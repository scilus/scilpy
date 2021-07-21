#!/usr/bin/env python
# encoding: utf-8

import argparse
import numpy as np
import nibabel as nib
import os

from scilpy.io.image import get_data_as_mask, get_data_as_label
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)

"""
Script to compute two masks of hard-to-track regions.

Crossings : Assigns value 2 if a voxel is part of more than one bundle and
nufo is >= 2.

Curves : Assigns value 3 if a voxel is part of more than one bundle and
nufo = 1.
"""


def buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description='Compute mask of hard-to-track '
                                'regions for multiresolution.')
    p.add_argument('in_first_bundle',
                   help='Input path to bundle')
    p.add_argument('in_second_bundle',
                   help='Input path to bundle')
    p.add_argument('in_numbers', nargs='+', type=int,
                   help='Numbers of bundles')
    p.add_argument('in_nufo',
                   help='Input path to nufo map')
    p.add_argument('out_mask',
                   help='Outpath for hard-to-track masks.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def compute_crossings_mask(img, first, second, nufo):
    # Create empty mask with bundle dimension
    mask_crossing = np.zeros(img.header.get_data_shape(), dtype=np.uint8)

    for i in range(mask_crossing.shape[-1]):
        mask_crossing[np.logical_and(first == 1, second == 1)] = 1
        mask_crossing[nufo < 2] = 0

    return mask_crossing


def compute_curves_mask(img, first, second, nufo):
    # Create empty mask with bundle dimension
    mask_curve = np.zeros(img.header.get_data_shape(), dtype=np.uint8)

    for i in range(mask_curve.shape[-1]):
        mask_curve[np.logical_and(first == 1, second == 1)] = 1
        mask_curve[np.logical_or(nufo == 0, nufo >= 2)] = 0

    return mask_curve


def main():

    parser = buildArgsParser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_first_bundle, args.in_second_bundle,
                        args.in_nufo])
    assert_outputs_exist(parser, args, args.out_mask)

    # Open bundle and nufo files
    first_img = nib.load(args.in_first_bundle)
    first_bundle = get_data_as_mask(first_img)

    second_img = nib.load(args.in_second_bundle)
    second_bundle = get_data_as_mask(second_img)

    nufo_img = nib.load(args.in_nufo)
    nufo = get_data_as_label(nufo_img)

    # Assign non-zero values for hard-to-track regions sections
    crossing_mask = compute_crossings_mask(nufo_img, first_bundle, second_bundle, nufo)
    curve_mask = compute_curves_mask(nufo_img, first_bundle, second_bundle, nufo)

    # Save crossing regions mask
    nib.save(nib.Nifti1Image(crossing_mask, nufo_img.affine), os.path.join(args.out_mask, 'mask_crossing_' + str(args.in_numbers[0]) + '_' + str(args.in_numbers[1]) + '.nii.gz'))
    nib.save(nib.Nifti1Image(curve_mask, nufo_img.affine), os.path.join(args.out_mask, 'mask_curve_' + str(args.in_numbers[0]) + '_' + str(args.in_numbers[1]) + '.nii.gz'))


if __name__ == "__main__":
    main()
