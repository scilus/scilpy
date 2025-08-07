#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to resample a dataset to match the resolution of another
reference dataset or to the resolution specified as in argument.

This script will reslice the volume to match the desired shape.

To:
    - pad or crop the volume to match the desired shape, use
      scil_volume_reshape.py.
    - reslice a volume to match the shape of another, use
      scil_volume_reslice_to_reference.py.
    - crop a volume to remove empty space, use scil_volume_crop.py.
Formerly: scil_resample_volume.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_verbose_arg, add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.image.volume_operations import resample_volume
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

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
    p.add_argument('--enforce_voxel_size', action='store_true',
                   help='Enforce --voxel_size even if there is a numerical'
                   ' difference after resampling.')
    p.add_argument('--enforce_dimensions', action='store_true',
                   help='Enforce the reference volume dimension.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Checking args
    assert_inputs_exist(parser, args.in_image, args.ref)
    assert_outputs_exist(parser, args, args.out_image)

    if args.enforce_dimensions and not args.ref:
        parser.error("Cannot enforce dimensions without a reference image.")

    if args.enforce_voxel_size and not args.voxel_size:
        parser.error("Cannot enforce voxel size without a voxel size.")

    if args.volume_size and (not len(args.volume_size) == 1 and
                             not len(args.volume_size) == 3):
        parser.error('Invalid dimensions for --volume_size.')

    if args.voxel_size and (not len(args.voxel_size) == 1 and
                            not len(args.voxel_size) == 3):
        parser.error('Invalid dimensions for --voxel_size.')

    logging.info('Loading raw data from %s', args.in_image)

    img = nib.load(args.in_image)

    ref_img = None
    if args.ref:
        ref_img = nib.load(args.ref)
        # Must not verify that headers are compatible. But can verify that, at
        # least, the last columns of their affines are compatible.
        if not np.array_equal(img.affine[:, -1], ref_img.affine[:, -1]):
            parser.error("The --ref image should have the same affine as the "
                         "input image (but with a different sampling).")

    # Resampling volume
    resampled_img = resample_volume(img, ref_img=ref_img,
                                    volume_shape=args.volume_size,
                                    iso_min=args.iso_min,
                                    voxel_res=args.voxel_size,
                                    interp=args.interp,
                                    enforce_dimensions=args.enforce_dimensions)

    # Saving results
    zooms = list(resampled_img.header.get_zooms())
    if args.voxel_size:
        if len(args.voxel_size) == 1:
            args.voxel_size = args.voxel_size * 3

        if not np.array_equal(zooms[:3], args.voxel_size):
            logging.warning('Voxel size is different from expected.'
                            ' Got: %s, expected: %s',
                            tuple(zooms), tuple(args.voxel_size))
            if args.enforce_voxel_size:
                logging.warning('Enforcing voxel size to %s',
                                tuple(args.voxel_size))
                zooms[0] = args.voxel_size[0]
                zooms[1] = args.voxel_size[1]
                zooms[2] = args.voxel_size[2]
                resampled_img.header.set_zooms(tuple(zooms))

    logging.info('Saving resampled data to %s', args.out_image)
    nib.save(resampled_img, args.out_image)


if __name__ == '__main__':
    main()
