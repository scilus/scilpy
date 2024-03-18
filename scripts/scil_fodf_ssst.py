#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute Constrained Spherical Deconvolution (CSD) fiber ODFs.

See [Tournier et al. NeuroImage 2007]

Formerly: scil_compute_ssst_fodf.py
"""

import argparse
import logging

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
import nibabel as nib
import numpy as np

from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              normalize_bvecs,
                                              is_normalized_bvecs)
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_b0_thresh_arg, add_overwrite_arg,
                             add_processes_arg, add_sh_basis_args,
                             add_skip_b0_check_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             parse_sh_basis_arg, assert_headers_compatible)
from scilpy.reconst.fodf import fit_from_model
from scilpy.reconst.sh import convert_sh_basis


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dwi',
                   help='Path of the input diffusion volume.')
    p.add_argument('in_bval',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('in_bvec',
                   help='Path of the bvecs file, in FSL format.')
    p.add_argument('frf_file',
                   help='Path of the FRF file')
    p.add_argument('out_fODF',
                   help='Output path for the fiber ODF coefficients.')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order used for the CSD. (Default: 8)')
    p.add_argument(
        '--mask', metavar='',
        help='Path to a binary mask. Only the data inside the mask will be '
             'used \nfor computations and reconstruction.')

    add_b0_thresh_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=True)
    add_sh_basis_args(p)
    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec,
                                 args.frf_file], args.mask)
    assert_outputs_exist(parser, args, args.out_fODF)
    assert_headers_compatible(parser, args.in_dwi, args.mask)

    # Loading data
    full_frf = np.loadtxt(args.frf_file)
    vol = nib.load(args.in_dwi)
    data = vol.get_fdata(dtype=np.float32)
    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    # Checking mask
    mask = get_data_as_mask(nib.load(args.mask),
                            dtype=bool) if args.mask else None

    sh_order = args.sh_order
    sh_basis, is_legacy = parse_sh_basis_arg(args)

    # Checking data and sh_order
    if data.shape[-1] < (sh_order + 1) * (sh_order + 2) / 2:
        logging.warning(
            'We recommend having at least {} unique DWI volumes, but you '
            'currently have {} volumes. Try lowering the parameter sh_order '
            'in case of non convergence.'.format(
                (sh_order + 1) * (sh_order + 2) / 2, data.shape[-1]))

    # Checking bvals, bvecs values and loading gtab
    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    # gtab.b0s_mask is used in dipy's csdeconv class.
    args.b0_threshold = check_b0_threshold(bvals.min(),
                                           b0_thr=args.b0_threshold,
                                           skip_b0_check=args.skip_b0_check)
    gtab = gradient_table(bvals, bvecs, b0_threshold=args.b0_threshold)

    # Checking full_frf and separating it
    if not full_frf.shape[0] == 4:
        raise ValueError('FRF file did not contain 4 elements. '
                         'Invalid or deprecated FRF format')
    frf = full_frf[0:3]
    mean_b0_val = full_frf[3]

    # Loading the sphere
    reg_sphere = get_sphere('symmetric362')

    # Computing CSD
    csd_model = ConstrainedSphericalDeconvModel(gtab, (frf, mean_b0_val),
                                                reg_sphere=reg_sphere,
                                                sh_order_max=sh_order)

    # Computing CSD fit
    csd_fit = fit_from_model(csd_model, data,
                             mask=mask, nbr_processes=args.nbr_processes)

    # Saving results
    shm_coeff = csd_fit.shm_coeff
    shm_coeff = convert_sh_basis(shm_coeff, reg_sphere, mask=mask,
                                 input_basis='descoteaux07',
                                 output_basis=sh_basis,
                                 is_input_legacy=True,
                                 is_output_legacy=is_legacy,
                                 nbr_processes=args.nbr_processes)
    nib.save(nib.Nifti1Image(shm_coeff.astype(np.float32),
                             vol.affine), args.out_fODF)


if __name__ == "__main__":
    main()
