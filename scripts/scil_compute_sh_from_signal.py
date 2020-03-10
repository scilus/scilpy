#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute the SH coefficient directly on the raw DWI signal.
"""

import argparse

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_force_b0_arg, add_overwrite_arg,
                             add_sh_basis_args, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.reconst.raw_signal import compute_sh_coefficients


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('dwi',
                   help='Path of the dwi volume.')
    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')
    p.add_argument('output',
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
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.dwi, args.bvals, args.bvecs])
    assert_outputs_exist(parser, args, args.output)

    vol = nib.load(args.dwi)
    dwi = vol.get_fdata()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    gtab = gradient_table(args.bvals, args.bvecs, b0_threshold=bvals.min())

    mask = None
    if args.mask:
        mask = nib.load(args.mask).get_fdata().astype(np.bool)

    sh = compute_sh_coefficients(dwi, gtab, args.sh_order, args.sh_basis,
                                 args.smooth,
                                 use_attenuation=args.use_attenuation,
                                 mask=mask)

    nib.save(nib.Nifti1Image(sh.astype(np.float32), vol.affine), args.output)


if __name__ == "__main__":
    main()
