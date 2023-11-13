#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply bias field correction to DWI. This script doesn't compute the bias
field itself. It ONLY applies an existing bias field. Use the ANTs
N4BiasFieldCorrection executable to compute the bias field
"""

import argparse

import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.dwi.operations import rescale_dwi


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='DWI Nifti image.')
    p.add_argument('in_bias_field',
                   help='Bias field Nifti image.')
    p.add_argument('out_name',
                   help='Corrected DWI Nifti image.')
    p.add_argument('--mask',
                   help='Apply bias field correction only in the region '
                        'defined by the mask.')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_dwi, args.in_bias_field], args.mask)
    assert_outputs_exist(parser, args, args.out_name)

    dwi_img = nib.load(args.in_dwi)
    dwi_data = dwi_img.get_fdata(dtype=np.float32)

    bias_field_img = nib.load(args.in_bias_field)
    bias_field_data = bias_field_img.get_fdata(dtype=np.float32)

    if args.mask:
        mask_img = nib.load(args.mask)
        nz_mask_data = np.nonzero(get_data_as_mask(mask_img))
    else:
        nz_mask_data = np.nonzero(np.average(dwi_data, axis=-1))

    nuc_dwi_data = np.divide(
        dwi_data[nz_mask_data],
        bias_field_data[nz_mask_data].reshape((len(nz_mask_data[0]), 1)))

    rescaled_nuc_data = rescale_dwi(dwi_data[nz_mask_data],
                                    nuc_dwi_data)

    dwi_data[nz_mask_data] = rescaled_nuc_data
    nib.save(nib.Nifti1Image(dwi_data, dwi_img.affine,
                             dwi_img.header),
             args.out_name)


if __name__ == "__main__":
    main()
