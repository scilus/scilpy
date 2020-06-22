#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute Constrained Spherical Deconvolution (CSD) fiber ODFs.

See [Tournier et al. NeuroImage 2007]
"""

import argparse
import logging

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_force_b0_arg,
                             add_sh_basis_args, add_processes_arg)
from scilpy.reconst.multi_processes import fit_from_model, convert_sh_basis
from scilpy.utils.bvec_bval_tools import (check_b0_threshold, normalize_bvecs,
                                          is_normalized_bvecs)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Path of the input diffusion volume.')
    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('bvecs',
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
             'used for computations and reconstruction.')

    add_force_b0_arg(p)
    add_sh_basis_args(p)
    add_processes_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.input, args.bvals, args.bvecs,
                                 args.frf_file])
    assert_outputs_exist(parser, args, args.out_fODF)

    # Loading data
    full_frf = np.loadtxt(args.frf_file)
    vol = nib.load(args.input)
    data = vol.get_fdata(dtype=np.float32)
    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    # Checking mask
    if args.mask is None:
        mask = None
    else:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")

    sh_order = args.sh_order

    # Checking data and sh_order
    check_b0_threshold(args.force_b0_threshold, bvals.min())
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
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    # Checking full_frf and separating it
    if not full_frf.shape[0] == 4:
        raise ValueError('FRF file did not contain 4 elements. '
                         'Invalid or deprecated FRF format')
    frf = full_frf[0:3]
    mean_b0_val = full_frf[3]

    # Loading the sphere
    reg_sphere = get_sphere('symmetric362')

    # Computing CSD
    csd_model = ConstrainedSphericalDeconvModel(
        gtab, (frf, mean_b0_val),
        reg_sphere=reg_sphere,
        sh_order=sh_order)

    # Computing CSD fit
    csd_fit = fit_from_model(csd_model, data,
                             mask=mask, nbr_processes=args.nbr_processes)

    # Saving results
    shm_coeff = csd_fit.shm_coeff
    if args.sh_basis == 'tournier07':
        shm_coeff = convert_sh_basis(shm_coeff, reg_sphere, mask=mask,
                                        nbr_processes=args.nbr_processes)
    nib.save(nib.Nifti1Image(shm_coeff.astype(np.float32),
                                vol.affine), args.out_fODF)

if __name__ == "__main__":
    main()
