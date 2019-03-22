#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute a single Fiber Response Function from a DWI.

A DTI fit is made, and voxels containing a single fiber population are
found using a threshold on the FA.
"""

from __future__ import division

from builtins import str
import argparse
import logging

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import auto_response
from dipy.segment.mask import applymask
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
    assert_outputs_exists, add_force_b0_arg)
from scilpy.utils.bvec_bval_tools import (
    check_b0_threshold, normalize_bvecs, is_normalized_bvecs)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="References: [1] Tournier et al. NeuroImage 2007")

    p.add_argument('input',
                   help='Path of the input diffusion volume.')
    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')
    p.add_argument('frf_file',
                   help='Path to the output FRF file, in .txt format, '
                        'saved by Numpy.')

    add_force_b0_arg(p)

    p.add_argument(
        '--mask',
        help='Path to a binary mask. Only the data inside the mask will be '
             'used for computations and reconstruction. Useful if no white '
             'matter mask is available.')
    p.add_argument(
        '--mask_wm', metavar='',
        help='Path to a binary white matter mask. Only the data inside this '
             'mask and above the threshold defined by --fa will be used to '
             'estimate the fiber response function.')
    p.add_argument(
        '--fa', dest='fa_thresh', default=0.7, type=float,
        help='If supplied, use this threshold as the initial threshold '
             'to select single fiber voxels. [%(default)s]')
    p.add_argument(
        '--min_fa', dest='min_fa_thresh', default=0.5, type=float,
        help='If supplied, this is the minimal value that will be tried '
             'when looking for single fiber voxels. [%(default)s]')
    p.add_argument(
        '--min_nvox', default=300, type=int,
        help='Minimal number of voxels needing to be identified as single '
             'fiber voxels in the automatic estimation. [%(default)s]')

    p.add_argument(
        '--roi_radius', default=10, type=int,
        help='If supplied, use this radius to select single fibers from the '
             'tensor to estimate the FRF. The roi will be a cube spanning '
             'from the middle of the volume in each direction. [%(default)s]')
    p.add_argument(
        '--roi_center', metavar='tuple(3)',
        help='If supplied, use this center to span the roi of size '
             'roi_radius. [center of the 3D volume]')

    add_overwrite_arg(p)

    p.add_argument('--verbose', '-v', action='store_true',
                   help='Produce verbose output.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.input, args.bvals, args.bvecs])
    assert_outputs_exists(parser, args, [args.frf_file])

    vol = nib.load(args.input)
    data = vol.get_data()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    check_b0_threshold(args, bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    if args.min_fa_thresh < 0.4:
        logging.warn(
            'Minimal FA threshold ({}) seems really small. Make sure it '
            'makes sense for this dataset.'.format(args.min_fa_thresh))

    if args.mask:
        mask = nib.load(args.mask).get_data().astype(np.bool)
        data = applymask(data, mask)

    if args.mask_wm:
        wm_mask = nib.load(args.mask_wm).get_data().astype('bool')
    else:
        wm_mask = np.ones_like(data[..., 0], dtype=np.bool)
        logging.warn(
            'No white matter mask specified! mask_data will be used instead, '
            'if it has been supplied. \nBe *VERY* careful about the '
            'estimation of the fiber response function to ensure no invalid '
            'voxel was used.')

    data_in_wm = applymask(data, wm_mask)

    fa_thresh = args.fa_thresh
    # Iteratively trying to fit at least 300 voxels. Lower the FA threshold
    # when it doesn't work. Fail if the fa threshold is smaller than
    # the min_threshold.
    # We use an epsilon since the -= 0.05 might incurs numerical imprecision.
    nvox = 0
    while nvox < args.min_nvox and fa_thresh >= args.min_fa_thresh - 0.00001:
        response, ratio, nvox = auto_response(gtab, data_in_wm,
                                              roi_center=args.roi_center,
                                              roi_radius=args.roi_radius,
                                              fa_thr=fa_thresh,
                                              return_number_of_voxels=True)

        logging.debug(
            'Number of indices is %s with threshold of %s', nvox, fa_thresh)
        fa_thresh -= 0.05

    if nvox < args.min_nvox:
        raise ValueError(
            "Could not find at least {} voxels with sufficient FA "
            "to estimate the FRF!".format(args.min_nvox))

    logging.debug("Found %i voxels with FA threshold %f for FRF estimation",
                  nvox, fa_thresh + 0.05)
    logging.debug("FRF eigenvalues: %s", str(response[0]))
    logging.debug("Ratio for smallest to largest eigen value is %f", ratio)
    logging.debug("Mean of the b=0 signal for voxels used for FRF: %f",
                  response[1])

    full_response = np.array([response[0][0], response[0][1],
                              response[0][2], response[1]])

    np.savetxt(args.frf_file, full_response)


if __name__ == "__main__":
    main()
