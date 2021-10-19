#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute powder average (mean diffusion weighted image) from a 
diffusion images.

By default will output an average image calculated from all images with
non-zero bvalue.

specify --bvalue to output an image for a single bvalue
"""

import argparse
import logging

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs

# Aliased to avoid clashes with images called mode.
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_force_b0_arg)
from scilpy.utils.bvec_bval_tools import (normalize_bvecs, is_normalized_bvecs,
                                          check_b0_threshold)
from scilpy.utils.filenames import add_filename_suffix, split_name_with_nii

logger = logging.getLogger("Compute_Powder_Average")
logger.setLevel(logging.INFO)


def _get_min_nonzero_signal(data):
    return np.min(data[data > 0])


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='Path of the input diffusion volume.')
    p.add_argument('in_bval',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('out_avg',
                   help='Path of the output file')
    
    add_overwrite_arg(p)
    p.add_argument(
        '--mask',
        help='Path to a binary mask.\nOnly data inside the mask will be used '
             'for powder avg. (Default: %(default)s)')

    p.add_argument()

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.not_all:
        args.fa = args.fa or 'fa.nii.gz'
        args.ga = args.ga or 'ga.nii.gz'
        args.rgb = args.rgb or 'rgb.nii.gz'
        args.md = args.md or 'md.nii.gz'
        args.ad = args.ad or 'ad.nii.gz'
        args.rd = args.rd or 'rd.nii.gz'
        args.mode = args.mode or 'mode.nii.gz'
        args.norm = args.norm or 'tensor_norm.nii.gz'
        args.tensor = args.tensor or 'tensor.nii.gz'
        args.evecs = args.evecs or 'tensor_evecs.nii.gz'
        args.evals = args.evals or 'tensor_evals.nii.gz'
        args.residual = args.residual or 'dti_residual.nii.gz'
        args.p_i_signal =\
            args.p_i_signal or 'physically_implausible_signals_mask.nii.gz'
        args.pulsation = args.pulsation or 'pulsation_and_misalignment.nii.gz'

    outputs = [args.fa, args.ga, args.rgb, args.md, args.ad, args.rd,
               args.mode, args.norm, args.tensor, args.evecs, args.evals,
               args.residual, args.p_i_signal, args.pulsation]
    if args.not_all and not any(outputs):
        parser.error('When using --not_all, you need to specify at least ' +
                     'one metric to output.')

    assert_inputs_exist(
        parser, [args.in_dwi, args.in_bval, args.in_bvec], args.mask)
    assert_outputs_exist(parser, args, outputs)

    img = nib.load(args.in_dwi)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    if args.mask is None:
        mask = None
    else:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)

    # Validate bvals and bvecs
    logging.info('Tensor estimation with the {} method...'.format(args.method))
    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    b0_thr = check_b0_threshold(
        args.force_b0_threshold, bvals.min(), bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)

    # Get tensors
    if args.method == 'restore':
        sigma = ne.estimate_sigma(data)
        tenmodel = TensorModel(gtab, fit_method=args.method, sigma=sigma,
                               min_signal=_get_min_nonzero_signal(data))
    else:
        tenmodel = TensorModel(gtab, fit_method=args.method,
                               min_signal=_get_min_nonzero_signal(data))

    tenfit = tenmodel.fit(data, mask)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    if args.tensor:
        # Get the Tensor values and format them for visualisation
        # in the Fibernavigator.
        tensor_vals = lower_triangular(tenfit.quadratic_form)
        correct_order = [0, 1, 3, 2, 4, 5]
        tensor_vals_reordered = tensor_vals[..., correct_order]
        fiber_tensors = nib.Nifti1Image(
            tensor_vals_reordered.astype(np.float32), affine)
        nib.save(fiber_tensors, args.tensor)

        del tensor_vals, fiber_tensors, tensor_vals_reordered

    if args.fa:
        fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
        nib.save(fa_img, args.fa)

        del fa_img




if __name__ == "__main__":
    main()
