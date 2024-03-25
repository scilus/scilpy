#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute multi-encoding multi-shell multi-tissue (memsmt)
Constrained Spherical Deconvolution ODFs.

In order to operate, the script only needs the data from one type of b-tensor
encoding. However, giving only a spherical one will not produce good fODFs, as
it only probes spherical shapes. As for planar encoding, it should technically
work alone, but seems to be very sensitive to noise and is yet to be properly
documented. We thus suggest to always use at least the linear encoding, which
will be equivalent to standard multi-shell multi-tissue if used alone, in
combinaison with other encodings. Note that custom encodings are not yet
supported, so that only the linear tensor encoding (LTE, b_delta = 1), the
planar tensor encoding (PTE, b_delta = -0.5), the spherical tensor encoding
(STE, b_delta = 0) and the cigar shape tensor encoding (b_delta = 0.5) are
available.

All of `--in_dwis`, `--in_bvals`, `--in_bvecs` and `--in_bdeltas` must have the
same number of arguments. Be sure to keep the same order of encodings
throughout all these inputs and to set `--in_bdeltas` accordingly (IMPORTANT).

By default, will output all possible files, using default names.
Specific names can be specified using the file flags specified in the
"File flags" section.

If --not_all is set, only the files specified explicitly by the flags
will be output.

>>> scil_fodf_memsmt.py wm_frf.txt gm_frf.txt csf_frf.txt --in_dwis LTE.nii.gz
    PTE.nii.gz STE.nii.gz --in_bvals LTE.bval PTE.bval STE.bval --in_bvecs
    LTE.bvec PTE.bvec STE.bvec --in_bdeltas 1 -0.5 0 --mask mask.nii.gz

Based on P. Karan et al., Bridging the gap between constrained spherical
deconvolution and diffusional variance decomposition via tensor-valued
diffusion MRI. Medical Image Analysis (2022)

Formerly: scil_compute_memsmt_fodf.py
"""

import argparse
import logging

from dipy.data import get_sphere
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
import nibabel as nib
import numpy as np

from scilpy.image.utils import extract_affine
from scilpy.io.btensor import (generate_btensor_input,
                               convert_bdelta_to_bshape)
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_sh_basis_args, add_skip_b0_check_arg,
                             add_tolerance_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             parse_sh_basis_arg, assert_headers_compatible)
from scilpy.reconst.fodf import (fit_from_model,
                                 verify_failed_voxels_shm_coeff,
                                 verify_frf_files)
from scilpy.reconst.sh import convert_sh_basis, verify_data_vs_sh_order


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_wm_frf',
                   help='Text file of WM response function.')
    p.add_argument('in_gm_frf',
                   help='Text file of GM response function.')
    p.add_argument('in_csf_frf',
                   help='Text file of CSF response function.')

    p.add_argument('--in_dwis', nargs='+', required=True,
                   help='Path to the input diffusion volume for each '
                        'b-tensor encoding type.')
    p.add_argument('--in_bvals', nargs='+', required=True,
                   help='Path to the bval file, in FSL format, for each '
                        'b-tensor encoding type.')
    p.add_argument('--in_bvecs', nargs='+', required=True,
                   help='Path to the bvec file, in FSL format, for each '
                        'b-tensor encoding type.')
    p.add_argument('--in_bdeltas', nargs='+', type=float,
                   choices=[0, 1, -0.5, 0.5], required=True,
                   help='Value of b_delta for each b-tensor encoding type, '
                        'in the same order as dwi, bval and bvec inputs.')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order used for the CSD. (Default: 8)')
    p.add_argument(
        '--mask',
        help='Path to a binary mask. Only the data inside the '
             'mask will be used for computations and reconstruction.')
    add_tolerance_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=False,
                          b0_tol_name='--tolerance')

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

    required = args.in_dwis + args.in_bvals + args.in_bvecs
    required += [args.in_wm_frf, args.in_gm_frf, args.in_csf_frf]
    assert_inputs_exist(parser, required, optional=args.mask)
    assert_outputs_exist(parser, args, arglist)
    assert_headers_compatible(parser, args.in_dwis, args.mask)

    if not (len(args.in_dwis) == len(args.in_bvals)
            == len(args.in_bvecs) == len(args.in_bdeltas)):
        parser.error("The number of given dwis, bvals, bvecs and bdeltas must "
                     "be the same. Please verify that all inputs were "
                     "correctly inserted.")

    # Loading data
    affine = extract_affine(args.in_dwis)

    wm_frf = np.loadtxt(args.in_wm_frf)
    gm_frf = np.loadtxt(args.in_gm_frf)
    csf_frf = np.loadtxt(args.in_csf_frf)

    # Note. This script does not currently allow using a separate b0_threshold
    # for the b0s. Using the tolerance. To change this, we would have to
    # change generate_btensor_input. Not doing any verification on the
    # bvals. Typically, we would use check_b0_threshold(bvals.min(), args)
    gtab, data, ubvals, ubdeltas = generate_btensor_input(
        args.in_dwis, args.in_bvals, args.in_bvecs, args.in_bdeltas,
        tol=args.tolerance, skip_b0_check=args.skip_b0_check)

    # Checking mask
    mask = get_data_as_mask(nib.load(args.mask),
                            dtype=bool) if args.mask else None

    # Checking data and sh_order
    verify_data_vs_sh_order(data, args.sh_order)
    sh_basis, is_legacy = parse_sh_basis_arg(args)

    # Checking response functions and computing mesmt response function
    wm_frf, gm_frf, csf_frf = verify_frf_files(wm_frf, gm_frf, csf_frf)

    # Loading spheres
    reg_sphere = get_sphere('symmetric362')

    # Starting main process!

    ubshapes = convert_bdelta_to_bshape(ubdeltas)
    memsmt_response = multi_shell_fiber_response(args.sh_order, ubvals,
                                                 wm_frf, gm_frf, csf_frf,
                                                 tol=args.tolerance,
                                                 btens=ubshapes)

    # Computing memsmt-CSD
    memsmt_model = MultiShellDeconvModel(gtab, memsmt_response,
                                         reg_sphere=reg_sphere,
                                         sh_order_max=args.sh_order)

    # Computing memsmt-CSD fit
    memsmt_fit = fit_from_model(memsmt_model, data,
                                mask=mask, nbr_processes=args.nbr_processes)

    # memsmt_fit is a MultiVoxelFit.
    #   - memsmt_fit.array_fit is a 3D np.ndarray, where value in each voxel is
    #     a dipy.reconst.mcsd.MSDeconvFit object.
    #   - When accessing memsmt_fit.all_shm_coeff, we get an array of shape
    #     (x, y, z, n), where n is the number of fitted values.
    shm_coeff = memsmt_fit.all_shm_coeff
    shm_coeff = verify_failed_voxels_shm_coeff(shm_coeff)

    vf = memsmt_fit.volume_fractions
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
                                 affine), args.wm_out_fODF)

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
                                 affine), args.gm_out_fODF)

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
                                 affine), args.csf_out_fODF)

    if args.vf:
        nib.save(nib.Nifti1Image(vf.astype(np.float32), affine), args.vf)

    if args.vf_rgb:
        vf_rgb = vf / np.max(vf) * 255
        vf_rgb = np.clip(vf_rgb, 0, 255)
        nib.save(nib.Nifti1Image(vf_rgb.astype(np.uint8),
                                 affine), args.vf_rgb)


if __name__ == "__main__":
    main()
