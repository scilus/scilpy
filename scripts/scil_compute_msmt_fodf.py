#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute Multishell Multi-tissue Constrained Spherical Deconvolution
ODFs.

By default, will output all possible files, using default names.
Specific names can be specified using the file flags specified in the
"File flags" section.

If --not_all is set, only the files specified explicitly by the flags
will be output.

Based on B. Jeurissen et al., Multi-tissue constrained spherical
deconvolution for improved analysis of multi-shell diffusion 
MRI data. Neuroimage (2014)
"""

import argparse
import logging

from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
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

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Path of the input diffusion volume.')
    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')
    p.add_argument('wm_frf',
                   help='Text file of WM response function.')
    p.add_argument('gm_frf',
                   help='Text file of GM response function.')
    p.add_argument('csf_frf',
                   help='Text file of CSF response function.')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order used for the CSD. (Default: 8)')
    p.add_argument(
        '--mask', metavar='',
        help='Path to a binary mask. Only the data inside the '
             'mask will be used for computations and reconstruction.')

    p.add_argument(
        '--not_all', action='store_true',
        help='If set, only saves the files specified using the '
             'file flags. (Default: False)')

    add_force_b0_arg(p)
    add_sh_basis_args(p)
    add_processes_arg(p)

    g = p.add_argument_group(title='File flags')

    g.add_argument(
        '--wm_fodf', metavar='file', default='',
        help='Output filename for the WM ODF coefficients.')
    g.add_argument(
        '--gm_fodf', metavar='file', default='',
        help='Output filename for the GM ODF coefficients.')
    g.add_argument(
        '--csf_fodf', metavar='file', default='',
        help='Output filename for the CSF ODF coefficients.')
    g.add_argument(
        '--vf', metavar='file', default='',
        help='Output filename for the volume fractions map.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.not_all:
        args.wm_fodf = args.wm_fodf or 'wm_fodf.nii.gz'
        args.gm_fodf = args.gm_fodf or 'gm_fodf.nii.gz'
        args.csf_fodf = args.csf_fodf or 'csf_fodf.nii.gz'
        args.vf = args.vf or 'vf.nii.gz'

    arglist = [args.wm_fodf, args.gm_fodf, args.csf_fodf, args.vf]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least ' +
                     'one file to output.')

    assert_inputs_exist(parser, [args.input, args.bvals, args.bvecs,
                                 args.wm_frf, args.gm_frf, args.csf_frf])
    assert_outputs_exist(parser, args, arglist)

    # Loading data
    wm_frf = np.loadtxt(args.wm_frf)
    gm_frf = np.loadtxt(args.gm_frf)
    csf_frf = np.loadtxt(args.csf_frf)
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
            'We recommend having at least {} unique DWIs volumes, but you '
            'currently have {} volumes. Try lowering the parameter --sh_order '
            'in case of non convergence.'.format(
                (sh_order + 1) * (sh_order + 2) / 2, data.shape[-1]))

    # Checking bvals, bvecs values and loading gtab
    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    # Checking response functions and computing msmt response function
    if not wm_frf.shape[1] == 4:
        raise ValueError('WM frf file did not contain 4 elements. '
                         'Invalid or deprecated FRF format')
    if not gm_frf.shape[1] == 4:
        raise ValueError('GM frf file did not contain 4 elements. '
                         'Invalid or deprecated FRF format')
    if not csf_frf.shape[1] == 4:
        raise ValueError('CSF frf file did not contain 4 elements. '
                         'Invalid or deprecated FRF format')
    msmt_response = multi_shell_fiber_response(sh_order,
                                               unique_bvals_tolerance(bvals, tol=20),
                                               wm_frf, gm_frf, csf_frf)
    
    # Loading spheres
    reg_sphere = get_sphere('symmetric362')

    # Computing msmt-CSD
    msmt_model = MultiShellDeconvModel(gtab, msmt_response,
                                       reg_sphere=reg_sphere,
                                       sh_order=sh_order)

    # Computing msmt-CSD fit
    msmt_fit = fit_from_model(msmt_model, data,
                              mask=mask, nbr_processes=args.nbr_processes)

    # Saving results
    if args.wm_fodf:
        shm_coeff = msmt_fit.shm_coeff
        if args.sh_basis == 'tournier07':
            shm_coeff = convert_sh_basis(shm_coeff, reg_sphere, mask=mask,
                                         nbr_processes=args.nbr_processes)
        nib.save(nib.Nifti1Image(shm_coeff.astype(np.float32),
                                    vol.affine), args.wm_fodf)

    if args.gm_fodf:
        shm_coeff = msmt_fit.all_shm_coeff[..., 1]
        if args.sh_basis == 'tournier07':
            shm_coeff = shm_coeff.reshape(shm_coeff.shape + (1,))
            shm_coeff = convert_sh_basis(shm_coeff, reg_sphere, mask=mask,
                                         nbr_processes=args.nbr_processes)
        nib.save(nib.Nifti1Image(shm_coeff.astype(np.float32),
                                    vol.affine), args.gm_fodf)

    if args.csf_fodf:
        shm_coeff = msmt_fit.all_shm_coeff[..., 0]
        if args.sh_basis == 'tournier07':
            shm_coeff = shm_coeff.reshape(shm_coeff.shape + (1,))
            shm_coeff = convert_sh_basis(shm_coeff, reg_sphere, mask=mask,
                                         nbr_processes=args.nbr_processes)
        nib.save(nib.Nifti1Image(shm_coeff.astype(np.float32),
                                    vol.affine), args.csf_fodf)

    if args.vf:
        nib.save(nib.Nifti1Image(msmt_fit.volume_fractions.astype(np.float32),
                                 vol.affine), args.vf)


if __name__ == "__main__":
    main()