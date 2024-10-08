#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to reshape a volume to match the resolution of another
reference volume or to the resolution specified as in argument. The resulting
volume will be centered in world space with respect to the reference volume or
the specified resolution.

This script will pad or crop the volume to match the desired shape.
To
    - interpolate/reslice to an arbitrary voxel size, use
      scil_volume_resample.py.
    - reslice a volume to match the shape of another, use
      scil_volume_reslice_to_reference.py.
    - crop a volume to constrain the field of view, use scil_volume_crop.py.
"""

import argparse
import logging

import nibabel as nib

from scilpy.io.utils import (add_verbose_arg, add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.image.volume_operations import reshape_volume


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

    p.add_argument(
        '--mode', default='constant',
        choices=['constant', 'edge', 'wrap', 'reflect'],
        help="Padding mode.\nconstant: pads with a constant value.\n"
             "edge: repeats the edge value.\nDefaults to [%(default)s].")
    p.add_argument('--constant_value', type=float, default=0,
                   help='Value to use for padding when mode is constant.')

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

    if args.volume_size and (not len(args.volume_size) == 1 and
                             not len(args.volume_size) == 3):
        parser.error('--volume_size takes in either 1 or 3 arguments.')

    logging.info('Loading raw data from %s', args.in_image)

    img = nib.load(args.in_image)

    ref_img = None
    if args.ref:
        ref_img = nib.load(args.ref)
        volume_shape = ref_img.shape[:3]
    else:
        if len(args.volume_size) == 1:
            volume_shape = [args.volume_size[0]] * 3
        else:
            volume_shape = args.volume_size

    # Resampling volume
    reshaped_img = reshape_volume(img, volume_shape,
                                  mode=args.mode,
                                  cval=args.constant_value)

    # Saving results
    logging.info('Saving reshaped data to %s', args.out_image)
    nib.save(reshaped_img, args.out_image)


if __name__ == '__main__':
    main()
