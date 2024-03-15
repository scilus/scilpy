#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a single Fiber Response Function from a DWI.

A DTI fit is made, and voxels containing a single fiber population are
found using a threshold on the FA.

Formerly: scil_compute_ssst_frf.py
"""

import argparse
import logging

from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.gradients.bvec_bval_tools import check_b0_threshold
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_b0_thresh_arg, add_overwrite_arg,
                             add_skip_b0_check_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             assert_roi_radii_format,
                             assert_headers_compatible)
from scilpy.reconst.frf import compute_ssst_frf


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="References: [1] Tournier et al. NeuroImage 2007")

    p.add_argument('in_dwi',
                   help='Path of the input diffusion volume.')
    p.add_argument('in_bval',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('in_bvec',
                   help='Path of the bvecs file, in FSL format.')
    p.add_argument('frf_file',
                   help='Path to the output FRF file, in .txt format, '
                        'saved by Numpy.')

    p.add_argument('--mask',
                   help='Path to a binary mask. Only the data inside the '
                        'mask will be used \nfor computations and '
                        'reconstruction. Useful if no white matter mask \n'
                        'is available.')
    p.add_argument('--mask_wm',
                   help='Path to a binary white matter mask. Only the data '
                        'inside this mask \nand above the threshold defined '
                        'by --fa_thresh will be used to estimate the \nfiber '
                        'response function.')
    p.add_argument('--fa_thresh', default=0.7, type=float,
                   help='If supplied, use this threshold as the initial '
                        'threshold to select \nsingle fiber voxels. '
                        '[%(default)s]')
    p.add_argument('--min_fa_thresh', default=0.5, type=float,
                   help='If supplied, this is the minimal value that will be '
                        'tried when looking \nfor single fiber '
                        'voxels. [%(default)s]')
    p.add_argument('--min_nvox', default=300, type=int,
                   help='Minimal number of voxels needing to be identified '
                        'as single fiber voxels \nin the automatic '
                        'estimation. [%(default)s]')

    p.add_argument('--roi_radii', default=[20], nargs='+', type=int,
                   help='If supplied, use those radii to select a cuboid roi '
                        'to estimate the \nresponse functions. The roi will '
                        'be a cuboid spanning from the middle of \nthe volume '
                        'in each direction with the different radii. The type '
                        'is either \nan int (e.g. --roi_radii 10) or an '
                        'array-like (3,) (e.g. --roi_radii 20 30 10). '
                        '[%(default)s]')
    p.add_argument('--roi_center', metavar='tuple(3)', nargs=3, type=int,
                   help='If supplied, use this center to span the roi of size '
                        'roi_radius. [center of the 3D volume]')

    add_b0_thresh_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=True)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec],
                        [args.mask, args.mask_wm])
    assert_outputs_exist(parser, args, args.frf_file)
    assert_headers_compatible(parser, args.in_dwi, [args.mask, args.mask_wm])

    roi_radii = assert_roi_radii_format(parser)

    vol = nib.load(args.in_dwi)
    data = vol.get_fdata(dtype=np.float32)

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    args.b0_threshold = check_b0_threshold(bvals.min(),
                                           b0_thr=args.b0_threshold,
                                           skip_b0_check=args.skip_b0_check)

    mask = get_data_as_mask(nib.load(args.mask),
                            dtype=bool) if args.mask else None
    mask_wm = get_data_as_mask(nib.load(args.mask_wm),
                               dtype=bool) if args.mask_wm else None

    full_response = compute_ssst_frf(
        data, bvals, bvecs, args.b0_threshold, mask=mask,
        mask_wm=mask_wm, fa_thresh=args.fa_thresh,
        min_fa_thresh=args.min_fa_thresh, min_nvox=args.min_nvox,
        roi_radii=roi_radii, roi_center=args.roi_center)

    np.savetxt(args.frf_file, full_response)


if __name__ == "__main__":
    main()
