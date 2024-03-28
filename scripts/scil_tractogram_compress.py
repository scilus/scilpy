#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compress tractogram by removing collinear (or almost) points.

The compression threshold represents the maximum distance (in mm) to the
original position of the point.

Formerly: scil_compress_streamlines.py
"""

import argparse
import logging

import nibabel as nib
from nibabel.streamlines import LazyTractogram
import numpy as np

from scilpy.io.streamlines import check_tracts_same_format
from scilpy.tractograms.tractogram_operations import \
    compress_streamlines_wrapper
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file (trk or tck).')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file (trk or tck).')

    p.add_argument('-e', dest='error_rate', type=float, default=0.1,
                   help='Maximum compression distance in mm [%(default)s].')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)
    check_tracts_same_format(parser, args.in_tractogram, args.out_tractogram)
    verify_compression_th(args.error_rate)

    in_tractogram = nib.streamlines.load(args.in_tractogram, lazy_load=True)
    compressed_streamlines = compress_streamlines_wrapper(in_tractogram,
                                                          args.error_rate)
    out_tractogram = LazyTractogram(compressed_streamlines,
                                    affine_to_rasmm=np.eye(4))
    nib.streamlines.save(out_tractogram, args.out_tractogram,
                         header=in_tractogram.header)


if __name__ == "__main__":
    main()
