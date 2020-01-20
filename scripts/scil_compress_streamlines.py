#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compress tractogram by removing collinear (or almost) points.

The compression threshold represents the maximum distance (in mm) to the
original position of the point.
"""

import argparse
import logging

from dipy.tracking.streamlinespeed import compress_streamlines
import nibabel as nib
from nibabel.streamlines import LazyTractogram
import numpy as np

from scilpy.io.streamlines import check_tracts_same_format
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file (trk or tck).')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file (trk or tck).')

    p.add_argument('-e', dest='error_rate', type=float, default=0.1,
                   help='Maximum compression distance in mm. '
                   '[default: %(default)s]')
    add_overwrite_arg(p)

    return p


def compress_streamlines_wrapper(tractogram, error_rate):
    return lambda: [(yield compress_streamlines(
        s, error_rate)) for s in tractogram.streamlines]


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)
    check_tracts_same_format(parser, args.in_tractogram, args.out_tractogram)

    if args.error_rate < 0.001 or args.error_rate > 1:
        logging.warning(
            'You are using an error rate of {}.\n'
            'We recommend setting it between 0.001 and 1.\n'
            '0.001 will do almost nothing to the streamlines\n'
            'while 1 will highly compress/linearize the streamlines'
            .format(args.error_rate))

    in_tractogram = nib.streamlines.load(args.in_tractogram, lazy_load=True)
    compressed_streamlines = compress_streamlines_wrapper(in_tractogram,
                                                          args.error_rate)

    out_tractogram = LazyTractogram(compressed_streamlines,
                                    affine_to_rasmm=np.eye(4))
    nib.streamlines.save(out_tractogram, args.out_tractogram,
                         header=in_tractogram.header)


if __name__ == "__main__":
    main()
