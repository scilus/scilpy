#!/usr/bin/env python
# encoding: utf-8

"""
Compute a density map from a streamlines file.

A specific value can be assigned instead of using the tract count.

This script correctly handles compressed streamlines.
"""

from __future__ import division

import argparse

import numpy as np
import nibabel as nb

from scilpy.io.streamlines import load_tracts_over_grid, check_tracts_support
from scilpy.io.utils import (
    add_overwrite_arg, add_tract_producer_arg,
    assert_inputs_exist, assert_outputs_exists)
from scilpy.tractanalysis import compute_robust_tract_counts_map


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'tracts', metavar='TRACTS',
        help='tract file name. name of the streamlines file, in a format '
             'supported by the Tractconverter.')
    p.add_argument(
        'ref_anat', metavar='REF_ANAT',
        help='path of the nifti file containing the reference anatomy, used '
             'for dimensions.')
    p.add_argument('out', metavar='OUTPUT_FILE',
                   help='path of the output image file.')

    p.add_argument(
        '--binary', metavar='FIXED_VALUE', type=int, nargs='?', const=1,
        help='if set, will store the same value for all intersected voxels, '
             'creating a binary map.\nWhen set without a value, 1 is used.\n'
             'If a value is given, will be used as the stored value.')
    add_tract_producer_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.tracts, args.ref_anat])
    assert_outputs_exists(parser, args, [args.out])
    check_tracts_support(parser, args.tracts, args.tracts_producer)

    max_ = np.iinfo(np.int16).max
    if args.binary is not None and (args.binary <= 0 or args.binary > max_):
        parser.error('The value of --binary ({}) '
                     'must be greater than 0 and smaller or equal to {}'
                     .format(args.binary, max_))

    streamlines = list(load_tracts_over_grid(
        args.tracts, args.ref_anat,
        start_at_corner=True, tract_producer=args.tracts_producer))

    # Compute weighting matrix taking the compression into account
    ref_img = nb.load(args.ref_anat)
    anat_dim = ref_img.get_header().get_data_shape()
    tract_counts = compute_robust_tract_counts_map(streamlines, anat_dim)

    if args.binary is not None:
        tract_counts[tract_counts > 0] = args.binary

    bin_img = nb.Nifti1Image(
        tract_counts.astype(np.int16), ref_img.get_affine())
    nb.save(bin_img, args.out)


if __name__ == "__main__":
    main()
