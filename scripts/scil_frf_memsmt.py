#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to estimate response functions for multi-encoding multi-shell
multi-tissue (memsmt) constrained spherical deconvolution. In order to operate,
the script only needs the data from one type of b-tensor encoding. However,
giving only a spherical one will not produce good fiber response functions, as
it only probes spherical shapes. As for planar encoding, it should technically
work alone, but seems to be very sensitive to noise and is yet to be properly
documented. We thus suggest to always use at least the linear encoding, which
will be equivalent to standard multi-shell multi-tissue if used alone, in
combinaison with other encodings. Note that custom encodings are not yet
supported, so that only the linear tensor encoding (LTE, b_delta = 1), the
planar tensor encoding (PTE, b_delta = -0.5), the spherical tensor encoding
(STE, b_delta = 0) and the cigar shape tensor encoding (b_delta = 0.5) are
available. Moreover, all of `--in_dwis`, `--in_bvals`, `--in_bvecs` and
`--in_bdeltas` must have the same number of arguments. Be sure to keep the
same order of encodings throughout all these inputs and to set `--in_bdeltas`
accordingly (IMPORTANT).

The script computes a response function for white-matter (wm),
gray-matter (gm), csf and the mean b=0.

In the wm, we compute the response function in each voxels where
the FA is superior at threshold_fa_wm.

In the gm (or csf), we compute the response function in each voxels where
the FA is below at threshold_fa_gm (or threshold_fa_csf) and where
the MD is below threshold_md_gm (or threshold_md_csf).

>>> scil_frf_memsmt.py wm_frf.txt gm_frf.txt csf_frf.txt --in_dwis LTE.nii.gz
    PTE.nii.gz STE.nii.gz --in_bvals LTE.bval PTE.bval STE.bval --in_bvecs
    LTE.bvec PTE.bvec STE.bvec --in_bdeltas 1 -0.5 0 --mask mask.nii.gz

Based on P. Karan et al., Bridging the gap between constrained spherical
deconvolution and diffusional variance decomposition via tensor-valued
diffusion MRI. Medical Image Analysis (2022)

Formerly: scil_compute_memsmt_frf.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.dwi.utils import extract_dwi_shell
from scilpy.image.utils import extract_affine
from scilpy.io.btensor import generate_btensor_input
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             assert_roi_radii_format, add_skip_b0_check_arg,
                             add_tolerance_arg,
                             assert_headers_compatible)
from scilpy.reconst.frf import compute_msmt_frf


def _build_arg_parser():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('out_wm_frf',
                   help='Path to the output WM frf file, in .txt format.')
    p.add_argument('out_gm_frf',
                   help='Path to the output GM frf file, in .txt format.')
    p.add_argument('out_csf_frf',
                   help='Path to the output CSF frf file, in .txt format.')

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
                        'in the same order as \ndwi, bval and bvec inputs.')

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

    p.add_argument('--fa_thr_wm',
                   default=0.7, type=float,
                   help='If supplied, use this threshold to select single WM '
                        'fiber voxels from \nthe FA inside the WM mask '
                        'defined by mask_wm. \nEach voxel above this '
                        'threshold will be selected. [%(default)s]')
    p.add_argument('--fa_thr_gm',
                   default=0.2, type=float,
                   help='If supplied, use this threshold to select GM voxels '
                        'from the FA inside \nthe GM mask defined by mask_gm. '
                        '\nEach voxel below this threshold will be selected. '
                        '[%(default)s]')
    p.add_argument('--fa_thr_csf',
                   default=0.1, type=float,
                   help='If supplied, use this threshold to select CSF voxels '
                        'from the FA inside \nthe CSF mask defined by '
                        'mask_csf. \nEach voxel below this threshold will be '
                        'selected. [%(default)s]')
    p.add_argument('--md_thr_gm',
                   default=0.0007, type=float,
                   help='If supplied, use this threshold to select GM voxels '
                        'from the MD inside \nthe GM mask defined by mask_gm. '
                        '\nEach voxel below this threshold will be selected. '
                        '[%(default)s]')
    p.add_argument('--md_thr_csf',
                   default=0.003, type=float,
                   help='If supplied, use this threshold to select CSF '
                        'voxels from the MD inside \nthe CSF mask defined by '
                        'mask_csf. \nEach voxel below this threshold will be '
                        'selected. [%(default)s]')

    p.add_argument('--min_nvox',
                   default=100, type=int,
                   help='Minimal number of voxels needed for each tissue masks'
                        ' in order to \nproceed to frf estimation. '
                        '[%(default)s]')
    add_tolerance_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=False,
                          b0_tol_name='--tolerance')
    p.add_argument('--dti_bval_limit',
                   type=int, default=1200,
                   help='The highest b-value taken for the DTI model. '
                        '[%(default)s]')
    p.add_argument('--roi_radii',
                   default=[20], nargs='+', type=int,
                   help='If supplied, use those radii to select a cuboid roi '
                        'to estimate the \nresponse functions. The roi will '
                        'be a cuboid spanning from the middle of \nthe volume '
                        'in each direction with the different radii. The type '
                        'is either \nan int (e.g. --roi_radii 10) or an '
                        'array-like (3,) (e.g. --roi_radii 20 30 10). '
                        '[%(default)s]')
    p.add_argument('--roi_center',
                   metavar='tuple(3)', nargs=3, type=int,
                   help='If supplied, use this center to span the cuboid roi '
                        'using roi_radii. \n[center of the 3D volume] '
                        '(e.g. --roi_center 66 79 79)')

    p.add_argument('--wm_frf_mask',
                   metavar='file', default='',
                   help='Path to the output WM frf mask file, the voxels used '
                        'to compute the WM frf.')
    p.add_argument('--gm_frf_mask',
                   metavar='file', default='',
                   help='Path to the output GM frf mask file, the voxels used '
                        'to compute the GM frf.')
    p.add_argument('--csf_frf_mask',
                   metavar='file', default='',
                   help='Path to the output CSF frf mask file, the voxels '
                        'used to compute the CSF frf.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    masks = [args.mask, args.mask_wm, args.mask_gm, args.mask_csf]
    assert_inputs_exist(parser, args.in_dwis + args.in_bvals + args.in_bvecs,
                        optional=masks)
    assert_outputs_exist(parser, args, [args.out_wm_frf, args.out_gm_frf,
                                        args.out_csf_frf])
    assert_headers_compatible(parser, args.in_dwis, masks)

    if not (len(args.in_dwis) == len(args.in_bvals)
            == len(args.in_bvecs) == len(args.in_bdeltas)):
        msg = """The number of given dwis, bvals, bvecs and bdeltas must be the
              same. Please verify that all inputs were correctly inserted."""
        raise ValueError(msg)

    affine = extract_affine(args.in_dwis)

    roi_radii = assert_roi_radii_format(parser)

    # Note. This script does not currently allow using a separate b0_threshold
    # for the b0s. Using the tolerance. To change this, we would have to
    # change generate_btensor_input. Not doing any verification on the
    # bvals. Typically, we would use check_b0_threshold(bvals.min(), args)
    gtab, data, ubvals, ubdeltas = generate_btensor_input(
        args.in_dwis, args.in_bvals, args.in_bvecs, args.in_bdeltas,
        tol=args.tolerance, skip_b0_check=args.skip_b0_check)

    if not np.all(ubvals <= args.dti_bval_limit):
        if np.sum(ubdeltas == 1) > 0:
            dti_ubvals = ubvals[ubdeltas == 1]
        elif np.sum(ubdeltas == -0.5) > 0:
            dti_ubvals = ubvals[ubdeltas == -0.5]
        elif np.sum(ubdeltas == args.in_bdelta_custom) > 0:
            dti_ubvals = ubvals[ubdeltas == args.in_bdelta_custom]
        else:
            raise ValueError("No encoding available for DTI.")
        vol = nib.Nifti1Image(data, affine)
        bvals_to_extract = dti_ubvals[dti_ubvals <= args.dti_bval_limit]
        indices_dti, data_dti, bvals_dti, bvecs_dti = \
            extract_dwi_shell(vol, gtab.bvals, gtab.bvecs,
                              bvals_to_extract, tol=1)

        bvals_dti = np.squeeze(bvals_dti)
        btens_dti = gtab.btens[indices_dti]
    else:
        data_dti = None
        bvals_dti = None
        bvecs_dti = None
        btens_dti = None

    mask = get_data_as_mask(nib.load(args.mask),
                            dtype=bool) if args.mask else None
    mask_wm = get_data_as_mask(nib.load(args.mask_wm),
                               dtype=bool) if args.mask_wm else None
    mask_gm = get_data_as_mask(nib.load(args.mask_gm),
                               dtype=bool) if args.mask_gm else None
    mask_csf = get_data_as_mask(nib.load(args.mask_csf),
                                dtype=bool) if args.mask_csf else None

    responses, frf_masks = compute_msmt_frf(data, gtab.bvals, gtab.bvecs,
                                            btens=gtab.btens,
                                            data_dti=data_dti,
                                            bvals_dti=bvals_dti,
                                            bvecs_dti=bvecs_dti,
                                            btens_dti=btens_dti,
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
                                            tol=0)

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
