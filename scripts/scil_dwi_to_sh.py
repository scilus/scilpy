#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute the SH coefficient directly on the raw DWI signal.

Formerly: scil_compute_sh_from_signal.py
"""

import argparse

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_force_b0_arg, add_overwrite_arg,
                             add_sh_basis_args, assert_inputs_exist,
                             add_verbose_arg, assert_outputs_exist)
from scilpy.reconst.sh import compute_sh_coefficients


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='Path of the dwi volume.')
    p.add_argument('in_bval',
                   help='Path of the b-value file, in FSL format.')
    p.add_argument('in_bvec',
                   help='Path of the b-vector file, in FSL format.')
    p.add_argument('out_sh',
                   help='Name of the output SH file to save.')

    p.add_argument('--sh_order', type=int, default=4,
                   help='SH order to fit (int). [%(default)s]')
    add_sh_basis_args(p)
    p.add_argument('--smooth', type=float, default=0.006,
                   help='Lambda-regularization coefficient in the SH fit '
                        '(float). [%(default)s]')
    p.add_argument('--use_attenuation', action='store_true',
                   help='If set, will use signal attenuation before fitting '
                        'the SH (i.e. divide by the b0).')
    add_force_b0_arg(p)
    p.add_argument('--mask',
                   help='Path to a binary mask.\nOnly data inside the mask '
                        'will be used for computations and reconstruction ')
    
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])
    assert_outputs_exist(parser, args, args.out_sh)

    vol = nib.load(args.in_dwi)
    dwi = vol.get_fdata(dtype=np.float32)

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    gtab = gradient_table(args.in_bval, args.in_bvec, b0_threshold=bvals.min())

    mask = None
    if args.mask:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)

    sh = compute_sh_coefficients(dwi, gtab, args.sh_order, args.sh_basis,
                                 args.smooth,
                                 use_attenuation=args.use_attenuation,
                                 mask=mask)

    nib.save(nib.Nifti1Image(sh.astype(np.float32), vol.affine), args.out_sh)


if __name__ == "__main__":
    main()
