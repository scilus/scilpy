#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply bias field correction to DWI. This script doesn't compute the bias
field itself. It ONLY applies an existing bias field. Use the ANTs
N4BiasFieldCorrection executable to compute the bias field
"""

from past.utils import old_div
import argparse

import nibabel as nib
import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exist)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('dwi', help='DWI Nifti image')
    parser.add_argument('bias_field', help='Bias field Nifti image')
    parser.add_argument('output', help='Corrected DWI Nifti image')
    parser.add_argument('--mask',
                        help='Apply bias field correction only in the region '
                             'defined by the mask')
    add_overwrite_arg(parser)
    return parser


def _rescale_intensity(val, slope, in_max, bc_max):
    return in_max - slope * (bc_max - val)


# https://github.com/stnava/ANTs/blob/master/Examples/N4BiasFieldCorrection.cxx
def _rescale_dwi(in_data, bc_data, mask_data=None):
    nz_in_data = in_data
    nz_bc_data = bc_data
    nz_mask_data = None

    if mask_data is not None:
        nz_mask_data = np.nonzero(mask_data)
        nz_in_data = in_data[nz_mask_data]
        nz_bc_data = bc_data[nz_mask_data]

    in_min = np.amin(nz_in_data)
    in_max = np.amax(nz_in_data)
    bc_min = np.amin(nz_bc_data)
    bc_max = np.amax(nz_bc_data)

    slope = old_div((in_max - in_min), (bc_max - bc_min))

    rescale_func = np.vectorize(_rescale_intensity, otypes=[np.float])
    rescaled_data = rescale_func(nz_bc_data, slope, in_max, bc_max)

    if mask_data is not None:
        bc_data[nz_mask_data] = rescaled_data
    else:
        bc_data = rescaled_data

    return bc_data


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.dwi, args.bias_field], args.mask)
    assert_outputs_exist(parser, args, args.output)

    dwi_img = nib.load(args.dwi)
    dwi_data = dwi_img.get_data()

    bias_field_img = nib.load(args.bias_field)
    bias_field_data = bias_field_img.get_data()

    mask_data = nib.load(args.mask).get_data() if args.mask else None
    nuc_dwi_data = np.divide(dwi_data, bias_field_data[..., np.newaxis])
    rescaled_nuc_data = _rescale_dwi(dwi_data, nuc_dwi_data, mask_data)

    nib.save(nib.Nifti1Image(rescaled_nuc_data, dwi_img.affine,
                             dwi_img.header),
             args.output)


if __name__ == "__main__":
    main()
