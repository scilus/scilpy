#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply bias field correction to DWI. This script doesn't compute the bias
field itself. It ONLY applies an existing bias field. Please use the ANTs
N4BiasFieldCorrection executable to compute the bias field.

Formerly: scil_apply_bias_field_on_dwi.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.dwi.operations import apply_bias_field
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             assert_headers_compatible)


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
                        'defined by the mask.\n'
                        'If this is not given, the bias field is still only '
                        'applied only in non-background data \n(i.e. where '
                        'the dwi is not 0).')
    
    add_verbose_arg(p)
    add_overwrite_arg(p)
    
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bias_field], args.mask)
    assert_outputs_exist(parser, args, args.out_name)
    assert_headers_compatible(parser, [args.in_dwi, args.in_bias_field],
                              args.mask)

    dwi_img = nib.load(args.in_dwi)
    dwi_data = dwi_img.get_fdata(dtype=np.float32)

    bias_field_img = nib.load(args.in_bias_field)
    bias_field_data = bias_field_img.get_fdata(dtype=np.float32)

    if args.mask:
        mask_data = get_data_as_mask(nib.load(args.mask))
    else:
        mask_data = np.average(dwi_data, axis=-1) != 0

    dwi_data = apply_bias_field(dwi_data, bias_field_data, mask_data)

    nib.save(nib.Nifti1Image(dwi_data, dwi_img.affine, dwi_img.header),
             args.out_name)


if __name__ == "__main__":
    main()
