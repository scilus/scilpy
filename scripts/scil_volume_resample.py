#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to resample a dataset to match the resolution of another
reference dataset or to the resolution specified as in argument.
"""

import argparse
import logging

import nibabel as nib

from scilpy.io.utils import (add_verbose_arg, add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.image.volume_operations import resample_volume


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_image',
                   help='Path of the input volume.')
    p.add_argument('out_image',
                   help='Path of the resampled volume.')

    res_group = p.add_mutually_exclusive_group(required=True)
    res_group.add_argument(
        '--ref',
        help='Reference volume to resample to.')
    res_group.add_argument(
        '--volume_size', nargs='+', type=int,
        help='Sets the size for the volume. If the value is set to is Y, '
             'it will resample to a shape of Y x Y x Y.')
    res_group.add_argument(
        '--voxel_size', nargs='+', type=float,
        help='Sets the voxel size. If the value is set to is Y, it will set '
             'a voxel size of Y x Y x Y.')
    res_group.add_argument(
        '--iso_min', action='store_true',
        help='Resample the volume to R x R x R with R being the smallest '
             'current voxel dimension.')

    p.add_argument(
        '--interp', default='lin',
        choices=['nn', 'lin', 'quad', 'cubic'],
        help="Interpolation mode.\nnn: nearest neighbour\nlin: linear\n"
             "quad: quadratic\ncubic: cubic\nDefaults to linear")
    p.add_argument('--enforce_dimensions', action='store_true',
                   help='Enforce the reference volume dimension.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Checking args
    assert_inputs_exist(parser, args.in_image, args.ref)
    assert_outputs_exist(parser, args, args.out_image)
    if args.enforce_dimensions and not args.ref:
        parser.error("Cannot enforce dimensions without a reference image")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.volume_size and (not len(args.volume_size) == 1 and
                             not len(args.volume_size) == 3):
        parser.error('Invalid dimensions for --volume_size.')

    if args.voxel_size and (not len(args.voxel_size) == 1 and
                            not len(args.voxel_size) == 3):
        parser.error('Invalid dimensions for --voxel_size.')

    logging.debug('Loading raw data from %s', args.in_image)

    img = nib.load(args.in_image)

    # Resampling volume
    resampled_img = resample_volume(img, ref=args.ref, res=args.volume_size,
                                    iso_min=args.iso_min, zoom=args.voxel_size,
                                    interp=args.interp,
                                    enforce_dimensions=args.enforce_dimensions)

    # Saving results
    logging.debug('Saving resampled data to %s', args.out_image)
    nib.save(resampled_img, args.out_image)


if __name__ == '__main__':
    main()
