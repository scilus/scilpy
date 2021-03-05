#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate and correct b-vectors using a fiber coherence index as presented
in Schilling et al, 2019. The script takes as input the principal direction
at each voxel, the b-vectors and the fractional anisotropy map and outputs
a corrected b-vectors file.

A typical pipeline could be:
>>> scil_compute_dti_metrics.py dwi.nii.gz bval bvec --not_all --fa fa.nii.gz
    --evecs peaks.nii.gz
>>> scil_correct_gradients.py bvec peaks_v1.nii.gz fa.nii.gz bvec_corr

Note that peaks_v1.nii.gz is the file containing the direction associated
to the highest eigen value at each voxel.

The output bvecs_corr file can then be used in future processing steps.
"""

import argparse
import logging
import numpy as np
import nibabel as nib

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.fiber_coherence import compute_fiber_coherence_table
from dipy.io.gradients import read_bvals_bvecs


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_bvecs', help='Path to bvecs file.')
    p.add_argument('in_peaks', help='Path to peaks file.')
    p.add_argument('in_fa', help='Path to the fractional anisotropy file.')
    p.add_argument('out_bvecs', help='Path to corrected bvecs file.')

    p.add_argument('--mask', help='Path to an optional mask.')
    p.add_argument('--fa_th', default=0.2, type=float,
                   help='FA threshold. Only voxels with FA higher '
                        'than fa_th will be considered.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    inputs = [args.in_bvecs, args.in_peaks, args.in_fa]
    if args.mask:
        inputs.append(args.mask)

    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, args.out_bvecs)

    _, bvecs = read_bvals_bvecs(None, args.in_bvecs)
    fa = nib.load(args.in_fa).get_fdata()
    peaks = nib.load(args.in_peaks).get_fdata()

    peaks = np.squeeze(peaks)
    if len(peaks.shape) > 4:
        parser.error('Peaks file contains more than one peak per voxel.')

    if args.mask:
        mask = get_data_as_mask(nib.load(args.mask))
        fa[np.logical_not(mask)] = 0
        peaks[np.logical_not(mask)] = 0

    peaks[fa < args.fa_th] = 0
    coherence, transform =\
        compute_fiber_coherence_table(peaks, fa)

    best_t = transform[np.argmax(coherence)]
    if (best_t == np.eye(3)).all():
        logging.info('b-vectors are already correct.')
        correct_bvecs = bvecs
    else:
        logging.info('Applying correction to b-vectors. '
                     'Transform is: \n{0}.'.format(best_t))
        correct_bvecs = np.dot(bvecs, best_t)

    logging.info('Saving bvecs to file: {0}.'.format(args.out_bvecs))
    np.savetxt(args.out_bvecs, correct_bvecs, "%.8f")


if __name__ == "__main__":
    main()
