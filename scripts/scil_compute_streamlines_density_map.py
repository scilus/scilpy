#!/usr/bin/env python
# encoding: utf-8

"""
Compute a density map from a streamlines file.

A specific value can be assigned instead of using the tract count.

This script correctly handles compressed streamlines.
"""
import argparse

import numpy as np
import nibabel as nib

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy.')
    p.add_argument('out_img',
                   help='path of the output image file.')

    p.add_argument('--binary', metavar='FIXED_VALUE', type=int,
                   nargs='?', const=1,
                   help='If set, will store the same value for all intersected'
                   ' voxels, creating a binary map.\n'
                   'When set without a value, 1 is used.\n'
                   'If a value is given, will be used as the stored value.')
    add_reference_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundle, optional=args.reference)
    assert_outputs_exist(parser, args, args.out_img)

    max_ = np.iinfo(np.int16).max
    if args.binary is not None and (args.binary <= 0 or args.binary > max_):
        parser.error('The value of --binary ({}) '
                     'must be greater than 0 and smaller or equal to {}'
                     .format(args.binary, max_))

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()
    streamlines = sft.streamlines
    transformation, dimensions, _, _ = sft.space_attributes

    streamline_count = compute_tract_counts_map(streamlines, dimensions)

    if args.binary is not None:
        streamline_count[streamline_count > 0] = args.binary

    nib.save(nib.Nifti1Image(streamline_count.astype(np.int16), transformation),
             args.out_img)


if __name__ == "__main__":
    main()
