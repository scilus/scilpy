#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to estimate response functions for multi-shell multi-tissue (MSMT) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
constrained spherical deconvolution.

The script computes a response function for white-matter (wm),
gray-matter (gm), csf and the mean b=0.

In the wm, we compute the response function in each voxels where
the FA is superior at threshold_fa_wm.

In the gm (or csf), we compute the response function in each voxels where
the FA is below at threshold_fa_gm (or threshold_fa_csf) and where
the MD is below threshold_md_gm (or threshold_md_csf).

Based on B. Jeurissen et al., Multi-tissue constrained spherical
deconvolution for improved analysis of multi-shell diffusion
MRI data. Neuroimage (2014)
"""

import argparse
import logging

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.mcsd import mask_for_response_msmt, response_from_mask_msmt
import nibabel as nib
import numpy as np
from pathlib import Path

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_force_b0_arg,
                             add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.utils.bvec_bval_tools import (check_b0_threshold, extract_dwi_shell,
                                          is_normalized_bvecs, normalize_bvecs)

from scilpy.reconst.b_tensor_utils import generate_btensor_input, extract_affine


def buildArgsParser():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('out_wm_frf',
                   help='Path to the output WM frf file, in .txt format.')
    p.add_argument('out_gm_frf',
                   help='Path to the output GM frf file, in .txt format.')
    p.add_argument('out_csf_frf',
                   help='Path to the output CSF frf file, in .txt format.')

    p.add_argument(
        '--in_dwi_linear', metavar='file', default=None,
        help='Path of the linear input diffusion volume.')
    p.add_argument(
        '--in_bval_linear', metavar='file', default=None,
        help='Path of the linear bvals file, in FSL format.')
    p.add_argument(
        '--in_bvec_linear', metavar='file', default=None,
        help='Path of the linear bvecs file, in FSL format.')
    p.add_argument(
        '--in_dwi_planar', metavar='file', default=None,
        help='Path of the planar input diffusion volume.')
    p.add_argument(
        '--in_bval_planar', metavar='file', default=None,
        help='Path of the planar bvals file, in FSL format.')
    p.add_argument(
        '--in_bvec_planar', metavar='file', default=None,
        help='Path of the planar bvecs file, in FSL format.')
    p.add_argument(
        '--in_dwi_spherical', metavar='file', default=None,
        help='Path of the spherical input diffusion volume.')
    p.add_argument(
        '--in_bval_spherical', metavar='file', default=None,
        help='Path of the spherical bvals file, in FSL format.')
    p.add_argument(
        '--in_bvec_spherical', metavar='file', default=None,
        help='Path of the spherical bvecs file, in FSL format.')
    p.add_argument(
        '--in_dwi_custom', metavar='file', default=None,
        help='Path of the custom input diffusion volume.')
    p.add_argument(
        '--in_bval_custom', metavar='file', default=None,
        help='Path of the custom bvals file, in FSL format.')
    p.add_argument(
        '--in_bvec_custom', metavar='file', default=None,
        help='Path of the custom bvecs file, in FSL format.')
    p.add_argument(
        '--in_bdelta_custom', type=float, choices=[0, 1, -0.5, 0.5],
        help='Value of the b_delta for the custom encoding.')
    p.add_argument(
        '--mask',
        help='Path to a binary mask. Only the data inside the mask will be '
             'used for computations and reconstruction. Useful if no tissue '
             'masks are available.')
    p.add_argument(
        '--mask_wm',
        help='Path to the input WM mask file, used to improve the'
             ' final WM frf mask.')
    p.add_argument(
        '--mask_gm',
        help='Path to the input GM mask file, used to improve the '
             'final GM frf mask.')
    p.add_argument(
        '--mask_csf',
        help='Path to the input CSF mask file, used to improve the'
             ' final CSF frf mask.')

    p.add_argument(
        '--thr_fa_wm', default=0.7, type=float,
        help='If supplied, use this threshold to select single WM fiber voxels '
             'from the FA inside the WM mask defined by wm_mask. Each voxel '
             'above this threshold will be selected. [%(default)s]')
    p.add_argument(
        '--thr_fa_gm', default=0.20, type=float,
        help='If supplied, use this threshold to select GM voxels from the FA '
             'inside the GM mask defined by gm_mask. Each voxel below this '
             'threshold will be selected. [%(default)s]')
    p.add_argument(
        '--thr_fa_csf', default=0.10, type=float,
        help='If supplied, use this threshold to select CSF voxels from the FA '
             'inside the CSF mask defined by csf_mask. Each voxel below this '
             'threshold will be selected. [%(default)s]')
    p.add_argument(
        '--thr_md_gm', default=0.0007, type=float,
        help='If supplied, use this threshold to select GM voxels from the MD '
             'inside the GM mask defined by gm_mask. Each voxel below this '
             'threshold will be selected. [%(default)s]')
    p.add_argument(
        '--thr_md_csf', default=0.003, type=float,
        help='If supplied, use this threshold to select CSF voxels from the MD '
             'inside the CSF mask defined by csf_mask. Each voxel below this '
             'threshold will be selected. [%(default)s]')

    p.add_argument(
        '--min_nvox', default=100, type=int,
        help='Minimal number of voxels needed for each tissue masks '
             'in order to proceed to frf estimation. [%(default)s]')                       
    p.add_argument(
        '--tolerance', type=int, default=20,
        help='The tolerated gap between the b-values to '
             'extract\nand the current b-value. [%(default)s]')
    p.add_argument(
        '--roi_radii', default=[20], nargs='+', type=int,
        help='If supplied, use those radii to select a cuboid roi '
             'to estimate the response functions. The roi will be '
             'a cuboid spanning from the middle of the volume in '
             'each direction with the different radii. The type is '
             'either an int or an array-like (3,). [%(default)s]')
    p.add_argument(
        '--roi_center', metavar='tuple(3)', nargs=3, type=int,
        help='If supplied, use this center to span the cuboid roi '
             'using roi_radii. [center of the 3D volume]')

    p.add_argument(
        '--wm_frf_mask', metavar='file', default='',
        help='Path to the output WM frf mask file, the voxels used '
             'to compute the WM frf.')
    p.add_argument(
        '--gm_frf_mask', metavar='file', default='',
        help='Path to the output GM frf mask file, the voxels used '
             'to compute the GM frf.')
    p.add_argument(
        '--csf_frf_mask', metavar='file', default='',
        help='Path to the output CSF frf mask file, the voxels used '
             'to compute the CSF frf.')

    p.add_argument(
        '--frf_table', metavar='file', default='',
        help='Path to the output frf table file. Saves the frf for '
             'each b-value, in .txt format.')

    add_force_b0_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [],
                    optional=[args.in_dwi_linear, args.in_bval_linear,
                                args.in_bvec_linear,
                                args.in_dwi_planar, args.in_bval_planar,
                                args.in_bvec_planar,
                                args.in_dwi_spherical, args.in_bval_spherical,
                                args.in_bvec_spherical])
    assert_outputs_exist(parser, args, [args.out_wm_frf, args.out_gm_frf,
                                        args.out_csf_frf])

    # Loading data
    input_files = [args.in_dwi_linear, args.in_dwi_planar,
                            args.in_dwi_spherical, args.in_dwi_custom]
    bvals_files = [args.in_bval_linear, args.in_bval_planar,
                           args.in_bval_spherical, args.in_bval_custom]
    bvecs_files = [args.in_bvec_linear, args.in_bvec_planar, 
                           args.in_bvec_spherical, args.in_bvec_custom]
    b_deltas_list = [1.0, -0.5, 0, args.in_bdelta_custom]

    gtab, data, ubvals, ubdeltas = generate_btensor_input(input_files,
                                                          bvals_files,
                                                          bvecs_files,
                                                          b_deltas_list,
                                                          args.force_b0_threshold,
                                                          tol=args.tolerance)

    affine = extract_affine(input_files)

    if len(args.roi_radii) == 1:
        roi_radii = args.roi_radii[0]
    elif len(args.roi_radii) == 2:
        parser.error('--roi_radii cannot be of size (2,).')
    else:
        roi_radii = args.roi_radii

    if not np.all(ubvals <= 1200):
        vol = nib.Nifti1Image(data, affine)
        outputs = extract_dwi_shell(vol, gtab.bvals, gtab.bvecs,
                                    ubvals[np.logical_and(ubvals <= 1200, ubdeltas == 1)], # !!!!!!!!!!!!!!!!!!!!!!
                                    tol=1)
        indices_dti, data_dti, bvals_dti, bvecs_dti = outputs
        gtab_dti = gradient_table(np.squeeze(bvals_dti), bvecs_dti,
                                  btens=gtab.btens[indices_dti])
    else:
        data_dti = data
        gtab_dti = gtab

    mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab_dti, data_dti,
                                                    roi_center=args.roi_center,
                                                    roi_radii=roi_radii,
                                                    wm_fa_thr=args.thr_fa_wm,
                                                    gm_fa_thr=args.thr_fa_gm,
                                                    csf_fa_thr=args.thr_fa_csf,
                                                    gm_md_thr=args.thr_md_gm,
                                                    csf_md_thr=args.thr_md_csf)

    if args.mask is not None:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
        mask_wm *= mask
        mask_gm *= mask
        mask_csf *= mask

    if args.mask_wm:
        tissue_mask_wm = get_data_as_mask(nib.load(args.mask_wm), dtype=bool)
        mask_wm *= tissue_mask_wm
    if args.mask_gm:
        tissue_mask_gm = get_data_as_mask(nib.load(args.mask_gm), dtype=bool)
        mask_gm *= tissue_mask_gm
    if args.mask_csf:
        tissue_mask_csf = get_data_as_mask(nib.load(args.mask_csf), dtype=bool)
        mask_csf *= tissue_mask_csf

    msg = """Could not find at least {0} voxels for the {1} mask. Look at
    previous warnings or be sure that external tissue masks overlap with the
    cuboid ROI."""
    min_nvox = args.min_nvox

    if np.sum(mask_wm) < min_nvox:
        raise ValueError(msg.format(min_nvox, "WM"))
    if np.sum(mask_gm) < min_nvox:
        raise ValueError(msg.format(min_nvox, "GM"))
    if np.sum(mask_csf) < min_nvox:
        raise ValueError(msg.format(min_nvox, "CSF"))

    masks = [mask_wm, mask_gm, mask_csf]
    mask_files = [args.wm_frf_mask, args.gm_frf_mask, args.csf_frf_mask]
    for mask, mask_file in zip(masks, mask_files):
        if mask_file:
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine),
                     mask_file)

    response_wm, response_gm, response_csf = response_from_mask_msmt(gtab,
                                                                     data,
                                                                     mask_wm,
                                                                     mask_gm,
                                                                     mask_csf,
                                                                     tol=0)

    frf_out = [args.out_wm_frf, args.out_gm_frf, args.out_csf_frf]
    responses = [response_wm, response_gm, response_csf]

    for frf, response in zip(frf_out, responses):
        np.savetxt(frf, response)
    
    if args.frf_table:
        if ubvals[0] < args.tolerance:
            bvals = ubvals[1:]
        else:
            bvals = ubvals
        iso_responses = np.concatenate((response_csf[:, :3],
                                        response_gm[:, :3]), axis=1)
        responses = np.concatenate((iso_responses, response_wm[:, :3]), axis=1)
        frf_table = np.vstack((bvals, responses.T)).T
        np.savetxt(args.frf_table, frf_table)



if __name__ == "__main__":
    main()
