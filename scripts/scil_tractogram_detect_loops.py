#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be used to remove loops in two types of streamline datasets:

  - Whole brain: For this type, the script removes streamlines if they
    make a loop with an angle of more than 360 degrees. It's possible to change
    this angle with the -a option. Warning: Don't use --qb option for a
    whole brain tractography.

  - Bundle dataset: For this type, it is possible to remove loops and
    streamlines outside of the bundle. For the sharp angle turn, use --qb
    option.

----------------------------------------------------------------------------
Reference:
QuickBundles based on [Garyfallidis12] Frontiers in Neuroscience, 2012.
----------------------------------------------------------------------------

Formerly: scil_detect_streamlines_loops.py
"""

import argparse
import json
import logging

from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_verbose_arg,
                             add_overwrite_arg,
                             add_processes_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             check_tracts_same_format,
                             validate_nbr_processes)
from scilpy.tractograms.tractogram_operations import filter_tractogram_data
from scilpy.tractograms.streamline_operations import \
    remove_loops_and_sharp_turns


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('in_tractogram',
                   help='Tractogram input file name.')
    p.add_argument('out_tractogram',
                   help='Output tractogram without loops.')
    p.add_argument('--looping_tractogram',
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
    p.add_argument('--display_counts', action='store_true',
                   help='Print streamline count before and after filtering')

    add_json_args(p)
    add_processes_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram,
                         optional=args.looping_tractogram)
    check_tracts_same_format(parser, [args.in_tractogram, args.out_tractogram,
                                      args.looping_tractogram])
    nbr_cpu = validate_nbr_processes(parser, args)

    if args.threshold <= 0:
        parser.error('Threshold "{}" '.format(args.threshold) +
                     'must be greater than 0')

    if args.angle <= 0:
        parser.error('Angle "{}" '.format(args.angle) +
                     'must be greater than 0')

    tractogram = load_tractogram_with_reference(
        parser, args, args.in_tractogram)

    streamlines = tractogram.streamlines

    ids_c = []

    ids_l = []

    if len(streamlines) > 1:
        ids_c = remove_loops_and_sharp_turns(
            streamlines, args.angle, use_qb=args.qb,
            qb_threshold=args.threshold,
            num_processes=nbr_cpu)
        ids_l = np.setdiff1d(np.arange(len(streamlines)), ids_c)
    else:
        parser.error(
            'Zero or one streamline in {}'.format(args.in_tractogram) +
            '. The file must have more than one streamline.')

    if len(ids_c) > 0:
        sft_c = filter_tractogram_data(tractogram, ids_c)
        save_tractogram(sft_c, args.out_tractogram)
    else:
        logging.warning(
            'No clean streamlines in {}'.format(args.in_tractogram))

    if args.display_counts:
        sc_bf = len(tractogram.streamlines)
        sc_af = len(sft_c.streamlines)
        print(json.dumps({'streamline_count_before_filtering': int(sc_bf),
                         'streamline_count_after_filtering': int(sc_af)},
                         indent=args.indent))

    if len(ids_l) == 0:
        logging.warning('No loops in {}'.format(args.in_tractogram))
    elif args.looping_tractogram:
        sft_l = filter_tractogram_data(tractogram, ids_l)
        save_tractogram(sft_l, args.looping_tractogram)


if __name__ == "__main__":
    main()
