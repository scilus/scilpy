#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute response functions for multi-shell multi-tissue (MSMT) constrained
spherical deconvolution from DWI data.

The script computes a response function for white-matter (wm),
gray-matter (gm), csf and the mean b=0.
    - In the wm, we compute the response function in each voxel where the FA is
      superior at threshold_fa_wm.
    - In the gm (or csf), we compute the response function in each voxel where
      the FA is below at threshold_fa_gm (or threshold_fa_csf) and where the MD
      is below threshold_md_gm (or threshold_md_csf).

We output one response function file for each tissue, containing the response
function for each b-value (arranged by lines). These are saved as the diagonal
of the axis-symmetric diffusion tensor (3 e-values) and a mean b0 value.
For example, a typical wm_frf is [15e-4, 4e-4, 4e-4, 700], where the tensor
e-values are (15,4,4)x10^-4 mm^2/s and the mean b0 is 700.

Based on B. Jeurissen et al., Multi-tissue constrained spherical deconvolution
for improved analysis of multi-shell diffusion MRI data. Neuroimage (2014)

Formerly: scil_compute_msmt_frf.py
"""

import argparse
import logging

from dipy.core.gradients import unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.dwi.utils import extract_dwi_shell
from scilpy.gradients.bvec_bval_tools import check_b0_threshold
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_skip_b0_check_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, assert_roi_radii_format,
                             assert_headers_compatible)
from scilpy.reconst.frf import compute_msmt_frf


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('in_dwi',
                   help='Path to the input diffusion volume.')
    p.add_argument('in_bval',
                   help='Path to the bval file, in FSL format.')
    p.add_argument('in_bvec',
                   help='Path to the bvec file, in FSL format.')
    p.add_argument('out_wm_frf',
                   help='Path to the output WM frf file, in .txt format.')
    p.add_argument('out_gm_frf',
                   help='Path to the output GM frf file, in .txt format.')
    p.add_argument('out_csf_frf',
                   help='Path to the output CSF frf file, in .txt format.')

    p.add_argument('--mask',
                   help='Path to a binary mask. Only the data inside the mask '
                        'will be used for\ncomputations and reconstruction. '
                        'Useful if no tissue masks are available.')
    p.add_argument('--mask_wm',
                   help='Path to the input WM mask file, used to improve the'
                        ' final WM frf mask.')
    p.add_argument('--mask_gm',
                   help='Path to the input GM mask file, used to improve the '
                        'final GM frf mask.')
    p.add_argument('--mask_csf',
                   help='Path to the input CSF mask file, used to improve the'
                        ' final CSF frf mask.')

    p.add_argument('--fa_thr_wm', default=0.7, type=float,
                   help='If supplied, use this threshold to select single WM '
                        'fiber voxels from the FA inside the WM mask defined '
                        ' by mask_wm. Each voxel above this threshold will '
                        'be selected. [%(default)s]')
    p.add_argument('--fa_thr_gm', default=0.2, type=float,
                   help='If supplied, use this threshold to select GM voxels '
                        'from the FA inside the GM mask defined by mask_gm. '
                        'Each voxel below this threshold will be selected.'
                        ' [%(default)s]')
    p.add_argument('--fa_thr_csf', default=0.1, type=float,
                   help='If supplied, use this threshold to select CSF voxels '
                        'from the FA inside the CSF mask defined by mask_csf. '
                        'Each voxel below this threshold will be selected. '
                        '[%(default)s]')
    p.add_argument('--md_thr_gm', default=0.0007, type=float,
                   help='If supplied, use this threshold to select GM voxels '
                        'from the MD inside the GM mask defined by mask_gm. '
                        'Each voxel below this threshold will be selected. '
                        '[%(default)s]')
    p.add_argument('--md_thr_csf', default=0.003, type=float,
                   help='If supplied, use this threshold to select CSF '
                        'voxels from the MD inside the CSF mask defined by '
                        'mask_csf. Each voxel below this threshold will be'
                        ' selected. [%(default)s]')

    p.add_argument('--min_nvox', default=100, type=int,
                   help='Minimal number of voxels needed for each tissue masks'
                        ' in order to proceed to frf estimation. '
                        '[%(default)s]')
    p.add_argument('--tolerance', type=int, default=20,
                   help='The tolerated gap between the b-values to '
                        'extract and the current b-value. [%(default)s]')
    add_skip_b0_check_arg(p, will_overwrite_with_min=False,
                          b0_tol_name='--tolerance')
    p.add_argument('--dti_bval_limit', type=int, default=1200,
                   help='The highest b-value taken for the DTI model. '
                        '[%(default)s]')
    p.add_argument('--roi_radii', default=[20], nargs='+', type=int,
                   help='If supplied, use those radii to select a cuboid roi '
                        'to estimate the response functions. The roi will be '
                        'a cuboid spanning from the middle of the volume in '
                        'each direction with the different radii. The type is '
                        'either an int (e.g. --roi_radii 10) or an array-like '
                        '(3,) (e.g. --roi_radii 20 30 10). [%(default)s]')
    p.add_argument('--roi_center', metavar='tuple(3)', nargs=3, type=int,
                   help='If supplied, use this center to span the cuboid roi '
                        'using roi_radii. [center of the 3D volume] '
                        '(e.g. --roi_center 66 79 79)')

    p.add_argument('--wm_frf_mask', metavar='file', default='',
                   help='Path to the output WM frf mask file, the voxels used '
                        'to compute the WM frf.')
    p.add_argument('--gm_frf_mask', metavar='file', default='',
                   help='Path to the output GM frf mask file, the voxels used '
                        'to compute the GM frf.')
    p.add_argument('--csf_frf_mask', metavar='file', default='',
                   help='Path to the output CSF frf mask file, the voxels '
                        'used to compute the CSF frf.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    masks = [args.mask, args.mask_wm, args.mask_gm, args.mask_csf]
    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec],
                        optional=masks)
    assert_outputs_exist(parser, args, [args.out_wm_frf, args.out_gm_frf,
                                        args.out_csf_frf])
    assert_headers_compatible(parser, args.in_dwi, optional=masks)

    roi_radii = assert_roi_radii_format(parser)

    # Loading
    vol = nib.load(args.in_dwi)
    data = vol.get_fdata(dtype=np.float32)
    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    dti_lim = args.dti_bval_limit

    # Note. This script does not currently allow using a separate b0_threshold
    # for the b0s. Using the tolerance. To fix this, we would need to change
    # the unique_bvals_tolerance and extract_dwi_shell methods.
    _ = check_b0_threshold(bvals.min(), b0_thr=args.tolerance,
                           skip_b0_check=args.skip_b0_check)
    list_bvals = unique_bvals_tolerance(bvals, tol=args.tolerance)
    if not np.all(list_bvals <= dti_lim):
        _, data_dti, bvals_dti, bvecs_dti = extract_dwi_shell(
            vol, bvals, bvecs, list_bvals[list_bvals <= dti_lim],
            tol=args.tolerance)
        bvals_dti = np.squeeze(bvals_dti)
    else:
        data_dti = None
        bvals_dti = None
        bvecs_dti = None

    mask = get_data_as_mask(nib.load(args.mask),
                            dtype=bool) if args.mask else None
    mask_wm = get_data_as_mask(nib.load(args.mask_wm),
                               dtype=bool) if args.mask_wm else None
    mask_gm = get_data_as_mask(nib.load(args.mask_gm),
                               dtype=bool) if args.mask_gm else None
    mask_csf = get_data_as_mask(nib.load(args.mask_csf),
                                dtype=bool) if args.mask_csf else None

    # Processing
    responses, frf_masks = compute_msmt_frf(data, bvals, bvecs,
                                            data_dti=data_dti,
                                            bvals_dti=bvals_dti,
                                            bvecs_dti=bvecs_dti,
                                            mask=mask, mask_wm=mask_wm,
                                            mask_gm=mask_gm, mask_csf=mask_csf,
                                            fa_thr_wm=args.fa_thr_wm,
                                            fa_thr_gm=args.fa_thr_gm,
                                            fa_thr_csf=args.fa_thr_csf,
                                            md_thr_gm=args.md_thr_gm,
                                            md_thr_csf=args.md_thr_csf,
                                            min_nvox=args.min_nvox,
                                            roi_radii=roi_radii,
                                            roi_center=args.roi_center,
                                            tol=args.tolerance)

    # Saving
    masks_files = [args.wm_frf_mask, args.gm_frf_mask, args.csf_frf_mask]
    for mask, mask_file in zip(frf_masks, masks_files):
        if mask_file:
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), vol.affine),
                     mask_file)

    frf_out = [args.out_wm_frf, args.out_gm_frf, args.out_csf_frf]

    for frf, response in zip(frf_out, responses):
        np.savetxt(frf, response)


if __name__ == "__main__":
    main()
