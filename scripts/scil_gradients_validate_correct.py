#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect sign flips and/or axes swaps in the gradients table from a fiber
coherence index [1]. The script takes as input the principal direction(s)
at each voxel, the b-vectors and the fractional anisotropy map and outputs
a corrected b-vectors file.

A typical pipeline could be:
>>> scil_dti_metrics.py dwi.nii.gz bval bvec --not_all --fa fa.nii.gz
    --evecs peaks.nii.gz
>>> scil_gradients_validate_correct.py bvec peaks_v1.nii.gz fa.nii.gz bvec_corr

Note that peaks_v1.nii.gz is the file containing the direction associated
to the highest eigenvalue at each voxel.

It is also possible to use a file containing multiple principal directions per
voxel, given that they are sorted by decreasing amplitude. In that case, the
first direction (with the highest amplitude) will be chosen for validation.
Only 4D data is supported, so the directions must be stored in a single
dimension. For example, peaks.nii.gz from scil_fodf_metrics.py could be used.

Formerly: scil_validate_and_correct_bvecs.py
"""

import argparse
import logging

from dipy.io.gradients import read_bvals_bvecs
import numpy as np
import nibabel as nib

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg,
                             assert_headers_compatible)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.fiber_coherence import compute_coherence_table_for_transforms


EPILOG = """
[1] Schilling KG, Yeh FC, Nath V, Hansen C, Williams O, Resnick S, Anderson AW,
Landman BA. A fiber coherence index for quality control of B-table orientation
in diffusion MRI scans. Magn Reson Imaging. 2019 May;58:82-89.
doi: 10.1016/j.mri.2019.01.018.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__, epilog=EPILOG)

    p.add_argument('in_bvec',
                   help='Path to bvec file.')
    p.add_argument('in_peaks',
                   help='Path to peaks file.')
    p.add_argument('in_FA',
                   help='Path to the fractional anisotropy file.')
    p.add_argument('out_bvec',
                   help='Path to corrected bvec file (FSL format).')

    p.add_argument('--mask',
                   help='Path to an optional mask. If set, FA and Peaks will '
                        'only be used inside the mask.')
    p.add_argument('--fa_threshold', default=0.2, type=float,
                   help='FA threshold. Only voxels with FA higher '
                        'than fa_threshold will be considered. [%(default)s]')
    p.add_argument('--column_wise', action='store_true',
                   help='Specify if input peaks are column-wise (..., 3, N) '
                        'instead of row-wise (..., N, 3).')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_bvec, args.in_peaks, args.in_FA],
                        optional=args.mask)
    assert_outputs_exist(parser, args, args.out_bvec)
    assert_headers_compatible(parser, [args.in_peaks, args.in_FA],
                              optional=args.mask)

    _, bvecs = read_bvals_bvecs(None, args.in_bvec)
    fa = nib.load(args.in_FA).get_fdata()
    peaks = nib.load(args.in_peaks).get_fdata()

    if peaks.shape[-1] > 3:
        logging.info('More than one principal direction per voxel was given.')
        peaks = peaks[..., 0:3]
        logging.info('The first peak is assumed to be the biggest.')

    # convert peaks to a volume of shape (H, W, D, N, 3)
    if args.column_wise:
        peaks = np.reshape(peaks, peaks.shape[:3] + (3, -1))
        peaks = np.transpose(peaks, axes=(0, 1, 2, 4, 3))
    else:
        peaks = np.reshape(peaks, peaks.shape[:3] + (-1, 3))

    peaks = np.squeeze(peaks)
    if args.mask:
        mask = get_data_as_mask(nib.load(args.mask), ref_shape=peaks.shape)
        fa[np.logical_not(mask)] = 0
        peaks[np.logical_not(mask)] = 0

    peaks[fa < args.fa_threshold] = 0
    coherence, transform = compute_coherence_table_for_transforms(peaks, fa)

    best_t = transform[np.argmax(coherence)]
    if (best_t == np.eye(3)).all():
        logging.info('b-vectors are already correct.')
        correct_bvecs = bvecs
    else:
        logging.info('Applying correction to b-vectors. '
                     'Transform is: \n{0}.'.format(best_t))
        correct_bvecs = np.dot(bvecs, best_t)

    logging.info('Saving bvecs to file: {0}.'.format(args.out_bvec))

    # FSL format (3, N)
    np.savetxt(args.out_bvec, correct_bvecs.T, '%.8f')


if __name__ == "__main__":
    main()
