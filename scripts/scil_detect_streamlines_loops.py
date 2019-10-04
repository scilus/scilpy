#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             check_tracts_same_format)
from scilpy.tractanalysis.features import remove_loops_and_sharp_turns


DESCRIPTION = """
This script can be used to remove loops in two types of streamline datasets:

  - Whole brain: For this type, the script removes streamlines if they
    make a loop with an angle of more than 360 degrees. It's possible to change
    this angle with the -a option. Warning: Don't use --qb option for a
    whole brain tractography.

  - Bundle dataset: For this type, it is possible to remove loops and
    streamlines outside of the bundle. For the sharp angle turn, use --qb option.

----------------------------------------------------------------------------
Reference:
QuickBundles based on [Garyfallidis12] Frontiers in Neuroscience, 2012.
----------------------------------------------------------------------------
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)
    p.add_argument('in_tractogram',
                   help='Tractogram input file name.')
    p.add_argument('out_tractogram',
                   help='Output tractogram without loops.')
    p.add_argument('--remaining_tractogram',
                   help='If set, saves detected looping streamlines.')
    p.add_argument('--qb', action='store_true',
                   help='If set, uses QuickBundles to detect\n' +
                        'outliers (loops, sharp angle turns).\n' +
                        'Should mainly be used with bundles. '
                        '[%(default)s]')
    p.add_argument('--threshold', default=8., type=float,
                   help='Maximal streamline to bundle distance\n' +
                        'for a streamline to be considered as\n' +
                        'a tracking error. [%(default)s]')
    p.add_argument('-a', dest='angle', default=360, type=float,
                   help='Maximum looping (or turning) angle of\n' +
                        'a streamline in degrees. [%(default)s]')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram,
                         optional=args.remaining_tractogram)
    check_tracts_same_format(parser, [args.in_tractogram, args.out_tractogram,
                                      args.remaining_tractogram])

    if args.threshold <= 0:
        parser.error('Threshold "{}" '.format(args.threshold) +
                     'must be greater than 0')

    if args.angle <= 0:
        parser.error('Angle "{}" '.format(args.angle) +
                     'must be greater than 0')

    tractogram = nib.streamlines.load(args.in_tractogram)
    streamlines = tractogram.streamlines

    streamlines_c = []
    loops = []
    if len(streamlines) > 1:
        streamlines_c, loops = remove_loops_and_sharp_turns(streamlines,
                                                            args.angle,
                                                            args.qb,
                                                            args.threshold)
    else:
        parser.error('Zero or one streamline in {}'.format(args.in_tractogram) +
                     '. The file must have more than one streamline.')

    if len(streamlines_c) > 0:
        tractogram_c = nib.streamlines.Tractogram(streamlines_c,
                                                  affine_to_rasmm=np.eye(4))
        nib.streamlines.save(tractogram_c, args.out_tractogram,
                             header=tractogram.header)
    else:
        logging.warning(
            'No clean streamlines in {}'.format(args.in_tractogram))

    if len(loops) == 0:
        logging.warning('No loops in {}'.format(args.in_tractogram))
    elif args.remaining_tractogram:
        tractogram_l = nib.streamlines.Tractogram(loops,
                                                  affine_to_rasmm=np.eye(4))
        nib.streamlines.save(tractogram_l, args.remaining_tractogram,
                             header=tractogram.header)


if __name__ == "__main__":
    main()
