#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse

import nibabel as nib
from nibabel.streamlines.tractogram import Tractogram
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tracking.tools import subsample_streamlines


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Keep only streamlines between [min, max] length',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('bundle',
                   help='Bundle to prune.')
    p.add_argument('pruned_bundle',
                   help='Pruned bundle.')

    p.add_argument('--min_length',
                   default=20., type=float,
                   help='Keep streamlines longer than min_length.' +
                        '[%(default)s]')
    p.add_argument('--max_length',
                   default=200., type=float,
                   help='Keep streamlines shorter than max_length. ' +
                        '[%(default)s]')
                        
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.bundle)
    assert_outputs_exist(parser, args, args.pruned_bundle)

    if args.min_length < 0:
        parser.error('--min_length {} should be at least 0'
                     .format(args.min_length))
    if args.max_length <= args.min_length:
        parser.error('--max_length {} should be greater than --min_length'
                     .format(args.max_length))

    tractogram = nib.streamlines.load(args.bundle)
    streamlines = tractogram.streamlines
    pruned_streamlines = subsample_streamlines(
        streamlines, args.min_length, args.max_length)

    if not pruned_streamlines:
        print("Pruning removed all the streamlines. Please adjust "
              "--{min,max}_length")
    else:
        pruned_tractogram = Tractogram(pruned_streamlines,
                                       affine_to_rasmm=np.eye(4))
        nib.streamlines.save(pruned_tractogram,
                             args.pruned_bundle,
                             header=tractogram.header)


if __name__ == '__main__':
    main()
