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

Formerly: scil_compute_msmt_fodf.py
"""

import argparse
import logging

from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
import nibabel as nib
import numpy as np

from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              normalize_bvecs,
                                              is_normalized_bvecs)
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_force_b0_arg,
                             add_sh_basis_args, add_processes_arg,
                             add_verbose_arg)
from scilpy.reconst.fodf import fit_from_model
from scilpy.reconst.sh import convert_sh_basis


def _build_arg_parser():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dwi',
                   help='Path of the input diffusion volume.')
    p.add_argument('in_bval',
                   help='Path of the bval file, in FSL format.')
    p.add_argument('in_bvec',
                   help='Path of the bvec file, in FSL format.')
    p.add_argument('in_wm_frf',
                   help='Text file of WM response function.')
    p.add_argument('in_gm_frf',
                   help='Text file of GM response function.')
    p.add_argument('in_csf_frf',
                   help='Text file of CSF response function.')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order used for the CSD. (Default: 8)')
    p.add_argument(
        '--mask', metavar='',
        help='Path to a binary mask. Only the data inside the '
             'mask will be used for computations and reconstruction.')
    p.add_argument(
        '--tolerance', type=int, default=20,
        help='The tolerated gap between the b-values to '
             'extract and the current b-value. [%(default)s]')

    add_force_b0_arg(p)
    add_sh_basis_args(p)
    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    p.add_argument(
        '--not_all', action='store_true',
        help='If set, only saves the files specified using the '
             'file flags. (Default: False)')

    g = p.add_argument_group(title='File flags')

    g.add_argument(
        '--wm_out_fODF', metavar='file', default='',
        help='Output filename for the WM fODF coefficients.')
    g.add_argument(
        '--gm_out_fODF', metavar='file', default='',
        help='Output filename for the GM fODF coefficients.')
    g.add_argument(
        '--csf_out_fODF', metavar='file', default='',
        help='Output filename for the CSF fODF coefficients.')
    g.add_argument(
        '--vf', metavar='file', default='',
        help='Output filename for the volume fractions map.')
    g.add_argument(
        '--vf_rgb', metavar='file', default='',
        help='Output filename for the volume fractions map in rgb.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    if not args.not_all:
        args.wm_out_fODF = args.wm_out_fODF or 'wm_fodf.nii.gz'
        args.gm_out_fODF = args.gm_out_fODF or 'gm_fodf.nii.gz'
        args.csf_out_fODF = args.csf_out_fODF or 'csf_fodf.nii.gz'
        args.vf = args.vf or 'vf.nii.gz'
        args.vf_rgb = args.vf_rgb or 'vf_rgb.nii.gz'

    arglist = [args.wm_out_fODF, args.gm_out_fODF, args.csf_out_fODF,
               args.vf, args.vf_rgb]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least ' +
                     'one file to output.')

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec,
                                 args.in_wm_frf, args.in_gm_frf,
                                 args.in_csf_frf])
    assert_outputs_exist(parser, args, arglist)

    # Loading data
    wm_frf = np.loadtxt(args.in_wm_frf)
    gm_frf = np.loadtxt(args.in_gm_frf)
    csf_frf = np.loadtxt(args.in_csf_frf)
    vol = nib.load(args.in_dwi)
    data = vol.get_fdata(dtype=np.float32)
    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    # Checking mask
    if args.mask is None:
        mask = None
    else:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")

    tol = args.tolerance
    sh_order = args.sh_order

    # Checking data and sh_order
    b0_thr = check_b0_threshold(
        args.force_b0_threshold, bvals.min(), bvals.min())

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
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)

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
    ubvals = unique_bvals_tolerance(bvals, tol=tol)
    msmt_response = multi_shell_fiber_response(sh_order, ubvals,
                                               wm_frf, gm_frf, csf_frf,
                                               tol=tol)

    # Loading spheres
    reg_sphere = get_sphere('symmetric362')

    # Computing msmt-CSD
    msmt_model = MultiShellDeconvModel(gtab, msmt_response,
                                       reg_sphere=reg_sphere,
                                       sh_order=sh_order)

    # Computing msmt-CSD fit
    msmt_fit = fit_from_model(msmt_model, data,
                              mask=mask, nbr_processes=args.nbr_processes)

    shm_coeff = msmt_fit.all_shm_coeff

    nan_count = len(np.argwhere(np.isnan(shm_coeff[..., 0])))
    voxel_count = np.prod(shm_coeff.shape[:-1])

    if nan_count / voxel_count >= 0.05:
        msg = """There are {} voxels out of {} that could not be solved by
        the solver, reaching a critical amount of voxels. Make sure to tune the
        response functions properly, as the solving process is very sensitive
        to it. Proceeding to fill the problematic voxels by 0.
        """
        logging.warning(msg.format(nan_count, voxel_count))
    elif nan_count > 0:
        msg = """There are {} voxels out of {} that could not be solved by
        the solver. Make sure to tune the response functions properly, as the
        solving process is very sensitive to it. Proceeding to fill the
        problematic voxels by 0.
        """
        logging.warning(msg.format(nan_count, voxel_count))

    shm_coeff = np.where(np.isnan(shm_coeff), 0, shm_coeff)

    vf = msmt_fit.volume_fractions
    vf = np.where(np.isnan(vf), 0, vf)

    # Saving results
    if args.wm_out_fODF:
        wm_coeff = shm_coeff[..., 2:]
        if args.sh_basis == 'tournier07':
            wm_coeff = convert_sh_basis(wm_coeff, reg_sphere, mask=mask,
                                        nbr_processes=args.nbr_processes)
        nib.save(nib.Nifti1Image(wm_coeff.astype(np.float32),
                                 vol.affine), args.wm_out_fODF)

    if args.gm_out_fODF:
        gm_coeff = shm_coeff[..., 1]
        if args.sh_basis == 'tournier07':
            gm_coeff = gm_coeff.reshape(gm_coeff.shape + (1,))
            gm_coeff = convert_sh_basis(gm_coeff, reg_sphere, mask=mask,
                                        nbr_processes=args.nbr_processes)
        nib.save(nib.Nifti1Image(gm_coeff.astype(np.float32),
                                 vol.affine), args.gm_out_fODF)

    if args.csf_out_fODF:
        csf_coeff = shm_coeff[..., 0]
        if args.sh_basis == 'tournier07':
            csf_coeff = csf_coeff.reshape(csf_coeff.shape + (1,))
            csf_coeff = convert_sh_basis(csf_coeff, reg_sphere, mask=mask,
                                         nbr_processes=args.nbr_processes)
        nib.save(nib.Nifti1Image(csf_coeff.astype(np.float32),
                                 vol.affine), args.csf_out_fODF)

    if args.vf:
        nib.save(nib.Nifti1Image(vf.astype(np.float32),
                                 vol.affine), args.vf)

    if args.vf_rgb:
        vf_rgb = vf / np.max(vf) * 255
        vf_rgb = np.clip(vf_rgb, 0, 255)
        nib.save(nib.Nifti1Image(vf_rgb.astype(np.uint8),
                                 vol.affine), args.vf_rgb)


if __name__ == "__main__":
    main()
