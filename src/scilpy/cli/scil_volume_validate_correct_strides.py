#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect when data strides are different from [1, 2, 3] and correct them.
The script takes as input a nifti file and outputs a nifti file with the
corrected strides if needed.

Input file can be 3D or 4D. Only the first 3 dimensions are considered
for the stride correction. In the case of DWI data, we recommand to also input
the b-values and b-vectors files to correct the b-vectors accordingly. If the
--validate_bvecs is set, the script first detects sign flips and/or axes swaps
in the b-vectors from a fiber coherence index [1] and corrects the b-vectors.
Then, the b-vectors are permuted and sign flipped to match the new strides.

A typical pipeline could be:
>>> scil_volume_validate_correct_strides t1.nii.gz t1_restride.nii.gz
>>> scil_volume_validate_correct_strides dwi.nii.gz dwi_restride.nii.gz
    --in_bvec dwi.bvec --out_bvec dwi_restride.bvec --validate_bvec
    --in_bval dwi.bval

------------------------------------------------------------------------------
Reference:
[1] Schilling KG, Yeh FC, Nath V, Hansen C, Williams O, Resnick S, Anderson AW,
    Landman BA. A fiber coherence index for quality control of B-table
    orientation in diffusion MRI scans. Magn Reson Imaging. 2019 May;58:82-89.
    doi: 10.1016/j.mri.2019.01.018.
------------------------------------------------------------------------------
"""

import argparse
import logging

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import TensorModel, fractional_anisotropy
import numpy as np
import nibabel as nib

from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              flip_gradient_sampling,
                                              is_normalized_bvecs,
                                              normalize_bvecs,
                                              swap_gradient_axis)
from scilpy.io.utils import (add_b0_thresh_arg, add_overwrite_arg,
                             add_skip_b0_check_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg)
from scilpy.reconst.fiber_coherence import compute_coherence_table_for_transforms
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_data', 
                   help='Path to input nifti file.')
    p.add_argument('out_data',
                   help='Path to output nifti file with corrected strides.')

    p.add_argument('--in_bvec',
                   help='Path to bvec file (FSL format). If provided, the '
                        'bvecs will \nbe permuted and sign flipped to match '
                        'the new strides.')
    p.add_argument('--out_bvec',
                   help='Path to output bvec file (FSL format). Must be '
                        'provided if --in_bvec is used.')
    p.add_argument('--validate_bvec', action='store_true',
                   help='If set, the script first detects sign flips and/or '
                        'axes swaps \nin the b-vectors from a fiber coherence '
                        'index [1] and corrects \nthe b-vectors before '
                        'permuting/sign flipping them to match the new '
                        'strides. \nIf not set, the b-vectors are only '
                        'permuted and sign flipped to match the new strides.')
    p.add_argument('--in_bval',
                   help='Path to bval file. Must be provided if '
                        '--validate_bvecs is used.')

    add_b0_thresh_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=True)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_data],
                        optional=[args.in_bvec, args.in_bval])
    assert_outputs_exist(parser, args, args.out_data, optional=args.out_bvec)

    if args.in_bvec and not args.out_bvec:
        parser.error('--out_bvec must be provided if --in_bvec is used.')
    if args.validate_bvec and (not args.in_bvec or not args.in_bval):
        parser.error('--in_bvec and --in_bval must be provided if '
                     '--validate_bvecs is set.')

    # Get the current strides
    img = nib.load(args.in_data)
    strides = nib.io_orientation(img.affine).astype(np.int8)
    strides = (strides[:, 0] + 1) * strides[:, 1]
    # Check if the strides are correct ([1, 2, 3])
    if np.array_equal(strides, [1, 2, 3]):
        is_stride_correct=True
        logging.warning('Input data already has the correct strides [1, 2, 3].'
                        ' No correction on data needed and outputed.')
    else:
        is_stride_correct=False
        logging.warning('Input data has strides {}. '
                        'Correcting to [1, 2, 3].'.format(strides))
        # Compute the required transform to get to [1, 2, 3]
        n = len(strides)
        transform = [0]*n
        for i, m in enumerate(strides):
            # Get the axis (0, 1, 2) and the sign of the current stride
            axis = abs(m) - 1
            sign = 1 if m > 0 else -1
            # Set the transform for this axis
            transform[axis] = sign * (i + 1)

        axes_to_flip = []
        swapped_order = []
        # Write the transform in a format compatible with the
        # flip_gradient_sampling and swap_gradient_axis functions (for bvecs)
        for next_axis in transform:
            if next_axis in [1, 2, 3]:
                swapped_order.append(next_axis - 1)
            elif next_axis in [-1, -2, -3]:
                axes_to_flip.append(abs(next_axis) - 1)
                swapped_order.append(abs(next_axis) - 1)

        # Write the transform in a format compatible with the nibabel
        # as_reoriented function (for image)
        ornt = np.column_stack((np.array(swapped_order, dtype=np.int8),
                                np.where(np.isin(range(n), axes_to_flip),
                                         -1, 1)))
        # Apply the transform to the image and save it
        new_img = img.as_reoriented(ornt)
        nib.save(new_img, args.out_data)

    if args.validate_bvec:
        logging.info('Validating b-vectors from fiber coherence index...')
        # Load and validate the data and bvals/bvecs
        data = img.get_fdata().astype(np.float32)
        if len(data.shape) != 4:
            parser.error('Input data must be DWI (4D) when --validate_bvec '
                         'is set.')
        bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
        if not is_normalized_bvecs(bvecs):
            logging.warning('Your b-vectors do not seem normalized...')
            bvecs = normalize_bvecs(bvecs)
        args.b0_threshold = check_b0_threshold(bvals.min(),
                                               b0_thr=args.b0_threshold,
                                               skip_b0_check=args.skip_b0_check)
        gtab = gradient_table(bvals, bvecs=bvecs,
                              b0_threshold=args.b0_threshold)
        tenmodel = TensorModel(gtab, fit_method='WLS',
                               min_signal=np.min(data[data > 0]))
        # Generate a mask to avoid fitting tensor on the whole image
        mask = np.zeros(data.shape[:3], dtype=bool)
        # Use a small cubic ROI at the center of the volume
        interval_i = slice(data.shape[0]//2,
                           data.shape[0]//2 + data.shape[0]//4)
        interval_j = slice(data.shape[1]//2,
                           data.shape[1]//2 + data.shape[1]//4)
        interval_k = slice(data.shape[2]//2,
                           data.shape[2]//2 + data.shape[2]//4)
        mask[interval_i, interval_j, interval_k] = 1
        # Compute the necessary DTI metrics to compute the coherence of bvecs
        tenfit = tenmodel.fit(data, mask=mask)
        fa = fractional_anisotropy(tenfit.evals)
        evecs = tenfit.evecs.astype(np.float32)[..., 0]
        evecs[fa < 0.2] = 0
        coherence, transform = compute_coherence_table_for_transforms(evecs, 
                                                                      fa)
        # Find the best transform and apply it to the bvecs if needed
        best_t = transform[np.argmax(coherence)]
        if (best_t == np.eye(3)).all():
            logging.info('The b-vectors are aligned with the original data.')
        else:
            logging.warning('Applying correction to b-vectors.')
            logging.info('Transform is: \n{0}.'.format(best_t))
            valid_bvecs = np.dot(bvecs, best_t)
            # If the data strides were correct, save the bvecs now
            if is_stride_correct:
                np.savetxt(args.out_bvec, valid_bvecs.T, "%.8f")

    # Apply the permutation and sign flip to the bvecs and save them
    if args.in_bvec and not is_stride_correct:
        if not args.validate_bvec:
            _, bvecs = read_bvals_bvecs(None, args.in_bvec)
        else:
            bvecs = valid_bvecs
        flipped_bvecs = flip_gradient_sampling(bvecs.T, axes_to_flip, 'fsl')
        swapped_flipped_bvecs = swap_gradient_axis(flipped_bvecs,
                                                   swapped_order, 'fsl')
        np.savetxt(args.out_bvec, swapped_flipped_bvecs, "%.8f")


if __name__ == "__main__":
    main()
