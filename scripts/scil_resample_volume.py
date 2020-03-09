#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to resample a dataset to match the resolution of another
reference dataset or to the resolution specified as in argument.
"""

import argparse
import logging

from scilpy.image.reslice import reslice
import nibabel as nib
import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exist)


def interp_code_to_order(interp_code):
    orders = {'nn': 0, 'lin': 1, 'quad': 2, 'cubic': 3}
    return orders[interp_code]


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input', action='store', metavar='in_vol', type=str,
                   help='Path of the input volume.')
    p.add_argument('output', action='store', metavar='out_vol', type=str,
                   help='Path of the resampled volume.')

    res_group = p.add_mutually_exclusive_group(required=True)
    res_group.add_argument('--ref', action='store', metavar='ref_vol',
                           help='Reference volume to resample to.')
    res_group.add_argument(
        '--resolution', action='store', metavar='float', type=float,
        help='Resolution to resample to. If the value it is set to is Y, it '
             'will resample to an isotropic resolution of Y x Y x Y.')
    res_group.add_argument(
        '--iso_min', action='store_true',
        help='Resample the volume to R x R x R with R being the smallest '
             'current voxel dimension ')

    p.add_argument(
        '--interp', action='store', default='lin', type=str,
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

    assert_inputs_exist(parser, args.input, args.ref)
    assert_outputs_exist(parser, args, args.output)
    if args.enforce_dimensions and not args.ref:
        parser.error("Cannot enforce dimensions without a reference image")

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug('Loading Raw data from %s', args.input)

    img = nib.load(args.input)
    data = img.get_data()
    affine = img.get_affine()
    original_zooms = img.get_header().get_zooms()[:3]

    if args.ref:
        ref_img = nib.load(args.ref)
        new_zooms = ref_img.header.get_zooms()[:3]
    elif args.resolution:
        new_zooms = [args.resolution] * 3
    elif args.iso_min:
        min_zoom = min(original_zooms)
        new_zooms = (min_zoom, min_zoom, min_zoom)

    logging.debug('Data shape: %s', data.shape)
    logging.debug('Data affine: %s', affine)
    logging.debug('Data affine setup: %s', nib.aff2axcodes(affine))
    logging.debug('Resampling data to %s with mode %s', new_zooms, args.interp)

    data2, affine2 = reslice(data, affine, original_zooms, new_zooms,
                             interp_code_to_order(args.interp))

    logging.debug('Resampled data shape: %s', data2.shape)
    logging.debug('Resampled data affine: %s', affine2)
    logging.debug('Resampled data affine setup: %s', nib.aff2axcodes(affine2))
    logging.debug('Saving resampled data to %s', args.output)

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


if __name__ == '__main__':
    main()
