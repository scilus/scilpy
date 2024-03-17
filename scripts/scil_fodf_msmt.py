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
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             add_sh_basis_args, add_skip_b0_check_arg,
                             add_verbose_arg, add_tolerance_arg,
                             parse_sh_basis_arg, assert_headers_compatible)
from scilpy.reconst.fodf import (fit_from_model,
                                 verify_failed_voxels_shm_coeff,
                                 verify_frf_files)
from scilpy.reconst.sh import convert_sh_basis, verify_data_vs_sh_order


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
    add_tolerance_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=False,
                          b0_tol_name='--tolerance')
    add_sh_basis_args(p)
    add_processes_arg(p)

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

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    if not args.not_all:
        args.wm_out_fODF = args.wm_out_fODF or 'wm_fodf.nii.gz'
        args.gm_out_fODF = args.gm_out_fODF or 'gm_fodf.nii.gz'
        args.csf_out_fODF = args.csf_out_fODF or 'csf_fodf.nii.gz'
        args.vf = args.vf or 'vf.nii.gz'
        args.vf_rgb = args.vf_rgb or 'vf_rgb.nii.gz'

    arglist = [args.wm_out_fODF, args.gm_out_fODF, args.csf_out_fODF,
               args.vf, args.vf_rgb]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least '
                     'one file to output.')

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec,
                                 args.in_wm_frf, args.in_gm_frf,
                                 args.in_csf_frf], args.mask)
    assert_outputs_exist(parser, args, arglist)
    assert_headers_compatible(parser, args.in_dwi, args.mask)

    # Loading data
    wm_frf = np.loadtxt(args.in_wm_frf)
    gm_frf = np.loadtxt(args.in_gm_frf)
    csf_frf = np.loadtxt(args.in_csf_frf)
    vol = nib.load(args.in_dwi)
    data = vol.get_fdata(dtype=np.float32)
    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    # Checking data and sh_order
    wm_frf, gm_frf, csf_frf = verify_frf_files(wm_frf, gm_frf, csf_frf)
    verify_data_vs_sh_order(data, args.sh_order)
    sh_basis, is_legacy = parse_sh_basis_arg(args)

    # Checking mask
    mask = get_data_as_mask(nib.load(args.mask),
                            dtype=bool) if args.mask else None

    # Checking bvals, bvecs values and loading gtab
    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    # Note. This script does not currently allow using a separate b0_threshold
    # for the b0s. Using the tolerance. To change this, we would have to
    # change many things in dipy. An issue has been added in dipy to
    # ask them to clarify the usage of gtab.b0s_mask. See here:
    #  https://github.com/dipy/dipy/issues/3015
    # b0_threshold option in gradient_table probably unused.
    _ = check_b0_threshold(bvals.min(), b0_thr=args.tolerance,
                           skip_b0_check=args.skip_b0_check)
    gtab = gradient_table(bvals, bvecs, b0_threshold=args.tolerance)

    # Loading spheres
    reg_sphere = get_sphere('symmetric362')

    # Starting main process!

    # Checking response functions and computing msmt response function
    ubvals = unique_bvals_tolerance(bvals, tol=args.tolerance)
    msmt_response = multi_shell_fiber_response(args.sh_order, ubvals,
                                               wm_frf, gm_frf, csf_frf,
                                               tol=args.tolerance)

    # Computing msmt-CSD
    msmt_model = MultiShellDeconvModel(gtab, msmt_response,
                                       reg_sphere=reg_sphere,
                                       sh_order_max=args.sh_order)

    # Computing msmt-CSD fit
    msmt_fit = fit_from_model(msmt_model, data,
                              mask=mask, nbr_processes=args.nbr_processes)

    # mmsmt_fit is a MultiVoxelFit.
    #   - memsmt_fit.array_fit is a 3D np.ndarray, where value in each voxel is
    #     a dipy.reconst.mcsd.MSDeconvFit object.
    #   - When accessing memsmt_fit.all_shm_coeff, we get an array of shape
    #     (x, y, z, n), where n is the number of fitted values.
    shm_coeff = msmt_fit.all_shm_coeff
    shm_coeff = verify_failed_voxels_shm_coeff(shm_coeff)

    vf = msmt_fit.volume_fractions
    vf = np.where(np.isnan(vf), 0, vf)

    # Saving results
    if args.wm_out_fODF:
        wm_coeff = shm_coeff[..., 2:]
        wm_coeff = convert_sh_basis(wm_coeff, reg_sphere, mask=mask,
                                    input_basis='descoteaux07',
                                    output_basis=sh_basis,
                                    is_input_legacy=True,
                                    is_output_legacy=is_legacy,
                                    nbr_processes=args.nbr_processes)
        nib.save(nib.Nifti1Image(wm_coeff.astype(np.float32),
                                 vol.affine), args.wm_out_fODF)

    if args.gm_out_fODF:
        gm_coeff = shm_coeff[..., 1]
        gm_coeff = gm_coeff.reshape(gm_coeff.shape + (1,))
        gm_coeff = convert_sh_basis(gm_coeff, reg_sphere, mask=mask,
                                    input_basis='descoteaux07',
                                    output_basis=sh_basis,
                                    is_input_legacy=True,
                                    is_output_legacy=is_legacy,
                                    nbr_processes=args.nbr_processes)
        nib.save(nib.Nifti1Image(gm_coeff.astype(np.float32),
                                 vol.affine), args.gm_out_fODF)

    if args.csf_out_fODF:
        csf_coeff = shm_coeff[..., 0]
        csf_coeff = csf_coeff.reshape(csf_coeff.shape + (1,))
        csf_coeff = convert_sh_basis(csf_coeff, reg_sphere, mask=mask,
                                     input_basis='descoteaux07',
                                     output_basis=sh_basis,
                                     is_input_legacy=True,
                                     is_output_legacy=is_legacy,
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
