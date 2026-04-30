#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect sign flips and/or axes swaps in the gradients table from a fiber
coherence index [1]. The script takes as input the DWI, b-values and b-vectors
and outputs a corrected b-vectors file.

A typical pipeline could be:
>>> scil_gradients_validate_correct dwi.nii.gz bval bvec bvec_corr

The script refits the DTI model 24 times (once for each possible axis
permutation and flip) and chooses the one that maximizes the fiber coherence
index. For performance, the fit is only performed on voxels with FA > 0.5.

------------------------------------------------------------------------------
Reference:
[1] Schilling KG, Yeh FC, Nath V, Hansen C, Williams O, Resnick S, Anderson AW,
    Landman BA. A fiber coherence index for quality control of B-table
    orientation in diffusion MRI scans. Magn Reson Imaging. 2019 May;58:82-89.
    doi: 10.1016/j.mri.2019.01.018.
------------------------------------------------------------------------------
"""

import argparse
import itertools
import logging

from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
import numpy as np
from tqdm import tqdm

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg,
                             add_b0_thresh_arg, add_skip_b0_check_arg)
from scilpy.io.image import get_data_as_mask
from scilpy.io.stateful_image import StatefulImage
from scilpy.gradients.bvec_bval_tools import check_b0_threshold
from scilpy.reconst.fiber_coherence import (compute_fiber_coherence,
                                            NB_FLIPS)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_dwi',
                   help='Path to the input DWI file.')
    p.add_argument('in_bval',
                   help='Path to the b-values file.')
    p.add_argument('in_bvec',
                   help='Path to the b-vectors file to validate.')
    p.add_argument('out_bvec',
                   help='Path to corrected bvec file (FSL format).')

    p.add_argument('--mask',
                   help='Path to an optional mask. If set, DTI fit will '
                        'only be performed inside the mask.')
    p.add_argument('--fa_threshold', default=0.5, type=float,
                   help='FA threshold. Only voxels with FA higher '
                        'than fa_threshold will be considered. [%(default)s]')

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
                        optional=args.mask)
    assert_outputs_exist(parser, args, args.out_bvec)

    # Loading data
    simg = StatefulImage.load(args.in_dwi)
    simg.load_gradients(args.in_bval, args.in_bvec)
    simg.to_ras()

    data = simg.get_fdata(dtype=np.float32)
    bvals = simg.bvals
    bvecs = simg.world_bvecs

    mask = None
    if args.mask:
        mask_simg = StatefulImage.load(args.mask)
        mask_simg.to_ras()
        mask = get_data_as_mask(mask_simg, dtype=bool)

    # Initial DTI fit to get FA and identify high-FA voxels
    args.b0_threshold = check_b0_threshold(bvals.min(),
                                           b0_thr=args.b0_threshold,
                                           skip_b0_check=args.skip_b0_check)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=args.b0_threshold)
    tenmodel = TensorModel(gtab, fit_method='WLS',
                           min_signal=np.min(data[data > 0]))
    tenfit = tenmodel.fit(data, mask=mask)
    fa = tenfit.fa

    # Define high-FA mask for coherence calculation
    high_fa_mask = fa > args.fa_threshold
    if mask is not None:
        high_fa_mask &= mask

    if np.sum(high_fa_mask) == 0:
        logging.error('No voxels found with FA > {}. Aborting.'
                      .format(args.fa_threshold))
        return

    # Generate 24 possible permutation/flips of gradient directions
    permutations = list(itertools.permutations([0, 1, 2]))
    transforms = np.zeros((len(permutations) * NB_FLIPS, 3, 3))
    for i in range(len(permutations)):
        transforms[i * NB_FLIPS, np.arange(3), permutations[i]] = 1
        for ii in range(3):
            flip = np.eye(3)
            flip[ii, ii] = -1
            transforms[ii + i * NB_FLIPS +
                       1] = transforms[i * NB_FLIPS].dot(flip)

    # Iterative refit and coherence calculation
    best_coherence = -1
    best_t = None

    logging.info('Refitting DTI 24 times for gradient validation...')
    for t in tqdm(transforms):
        # Transform bvecs
        # Note: Dipy expects bvecs as (N, 3). We apply the transform to axes.
        # G' = G @ T
        bvecs_candidate = bvecs @ t

        gtab_candidate = gradient_table(bvals, bvecs=bvecs_candidate,
                                        b0_threshold=args.b0_threshold)
        tenmodel_candidate = TensorModel(gtab_candidate, fit_method='WLS',
                                         min_signal=np.min(data[data > 0]))

        # Fit ONLY on the high-FA mask to save time
        tenfit_candidate = tenmodel_candidate.fit(data, mask=high_fa_mask)

        # Extract the principal direction (v1)
        # evecs is (H, W, D, 3, 3), evecs[..., 0] is the first eigenvector (peak)
        peaks = tenfit_candidate.evecs[..., 0]

        # Compute coherence
        coherence = compute_fiber_coherence(peaks, fa)

        if coherence > best_coherence:
            best_coherence = coherence
            best_t = t

    if (best_t == np.eye(3)).all():
        logging.info('b-vectors are already correct. Coherence: {:.2f}'
                     .format(best_coherence))
        correct_bvecs = bvecs
    else:
        logging.info('Applying correction to b-vectors. Coherence: {:.2f} '
                     '\nTransform is: \n{}.'.format(best_coherence, best_t))
        correct_bvecs = bvecs @ best_t

    logging.info('Saving bvecs to file: {0}.'.format(args.out_bvec))

    # Save using StatefulImage to ensure they are in the original voxel space
    simg.attach_gradients(bvals, correct_bvecs, original_order=False)
    simg.save_gradients(args.in_bval, args.out_bvec)


if __name__ == "__main__":
    main()
