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

    # Checking args
    assert_inputs_exist(parser, args.input, args.ref)
    assert_outputs_exist(parser, args, args.output)
    if args.enforce_dimensions and not args.ref:
        parser.error("Cannot enforce dimensions without a reference image")

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug('Loading Raw data from %s', args.input)

    img = nib.load(args.input)

    # Resampling volume
    resampled_img = resample_volume(img, ref=args.ref, res=args.resolution,
                                    iso_min=args.iso_min, interp=args.interp,
                                    enforce_dimensions=args.enforce_dimensions)

    # Saving results
    logging.debug('Saving resampled data to %s', args.output)
    nib.save(resampled_img, args.output)


if __name__ == '__main__':
    main()
