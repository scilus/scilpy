#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to resample a dataset to match the resolution of another
reference dataset or to the resolution specified as in argument.
"""

import argparse
import logging

import nibabel as nib

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exist)
from scilpy.image.resample_volume import resample_volume


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input', metavar='in_vol',
                   help='Path of the input volume.')
    p.add_argument('output', metavar='out_vol',
                   help='Path of the resampled volume.')

    res_group = p.add_mutually_exclusive_group(required=True)
    res_group.add_argument('--ref', metavar='ref_vol',
                           help='Reference volume to resample to.')
    res_group.add_argument(
        '--resolution', metavar='float', type=float,
        help='Resolution to resample to. If the value it is set to is Y, it '
             'will resample to an isotropic resolution of Y x Y x Y.')
    res_group.add_argument(
        '--iso_min', action='store_true',
        help='Resample the volume to R x R x R with R being the smallest '
             'current voxel dimension ')

    p.add_argument(
        '--interp', default='lin',
        choices=['nn', 'lin', 'quad', 'cubic'],
        help="Interpolation mode.\nnn: nearest neighbour\nlin: linear\n"
             "quad: quadratic\ncubic: cubic\nDefaults to linear")
    p.add_argument('--enforce_dimensions', action='store_true',
                   help='Enforce the reference volume dimension.')
    add_overwrite_arg(p)
    p.add_argument('-v', action='store_true', dest='verbose',
                   help='Use verbose output. Default: false.')

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    # Checking args
    assert_inputs_exist(parser, args.input, args.ref)
    assert_outputs_exist(parser, args, args.output)
    if args.enforce_dimensions and not args.ref:
        parser.error("Cannot enforce dimensions without a reference image")

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug('Loading Raw data from {}'.format(args.input))

    img = nib.load(args.input)

    # Resampling volume
    if args.ref:
        resampled_image = resample_volume(img, ref=args.ref,
                                          interp=args.interp,
                                          enforce_dimensions=args.enforce_dimensions)
    elif args.resolution:
        resampled_image = resample_volume(img, res=args.resolution,
                                          interp=args.interp)
    elif args.iso_min:
        min_zoom = min(original_zooms)
        new_zooms = (min_zoom, min_zoom, min_zoom)

    logging.debug('Data shape: {}'.format(data.shape))
    logging.debug('Data affine: {}'.format(affine))
    logging.debug('Data affine setup: {}'.format(nib.aff2axcodes(affine)))
    logging.debug('Resampling data to {} with mode {}'.format(
                  new_zooms, args.interp))

    data2, affine2 = reslice(data, affine, original_zooms, new_zooms,
                             interp_code_to_order(args.interp))

    logging.debug('Resampled data shape: {}'.format(data2.shape))
    logging.debug('Resampled data affine: {}'.format(affine2))
    logging.debug('Resampled data affine setup: {}'.format(
                  nib.aff2axcodes(affine2)))
    logging.debug('Saving resampled data to {}'.format(args.output))

    if args.enforce_dimensions:
        computed_dims = data2.shape
        ref_dims = ref_img.shape[:3]
        if computed_dims != ref_dims:
            fix_dim_volume = np.zeros(ref_dims)
            x_dim = min(computed_dims[0], ref_dims[0])
            y_dim = min(computed_dims[1], ref_dims[1])
            z_dim = min(computed_dims[2], ref_dims[2])

            fix_dim_volume[:x_dim, :y_dim, :z_dim] = \
                data2[:x_dim, :y_dim, :z_dim]
            data2 = fix_dim_volume

    nib.save(nib.Nifti1Image(data2, affine2), args.output)
>>>>>>> f4921fc6165dc25862d11c380961ac363d30b7ed


if __name__ == '__main__':
    main()
