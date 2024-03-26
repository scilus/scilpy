#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a density map from a streamlines file. Can be binary.

This script correctly handles compressed streamlines.

Formerly: scil_compute_streamlines_density_map.py
"""
import argparse
import logging

import numpy as np
import nibabel as nib

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist, add_verbose_arg, 
                             assert_outputs_exist)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Tractogram filename.')
    p.add_argument('out_img',
                   help='path of the output image file.')

    p.add_argument('--binary', metavar='FIXED_VALUE', type=int,
                   nargs='?', const=1,
                   help='If set, will store the same value for all '
                        'intersected voxels, \ncreating a binary map.'
                        'When set without a value, 1 is used (and dtype \n'
                        'uint8). If a value is given, will be used as the '
                        'stored value.')
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, args.in_bundle, optional=args.reference)
    assert_outputs_exist(parser, args, args.out_img)

    max_ = np.iinfo(np.int16).max
    if args.binary is not None and (args.binary <= 0 or args.binary > max_):
        parser.error('The value of --binary ({}) '
                     'must be greater than 0 and smaller or equal to {}'
                     .format(args.binary, max_))

    # Loading
    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()
    transformation, dimensions, _, _ = sft.space_attributes

    # Processing
    streamline_count = compute_tract_counts_map(sft.streamlines, dimensions)

    # Saving
    dtype_to_use = np.int32
    if args.binary is not None:
        if args.binary == 1:
            dtype_to_use = np.uint8
        streamline_count[streamline_count > 0] = args.binary

    img = nib.Nifti1Image(streamline_count.astype(dtype_to_use),
                          transformation)
    nib.save(img, args.out_img)


if __name__ == "__main__":
    main()
