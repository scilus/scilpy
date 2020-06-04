#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract B0s from DWI.

The default behavior is to save the first b0 of the series.
"""

import argparse
import logging
import os

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs

import nibabel as nib
import numpy as np

from scilpy.io.utils import add_verbose_arg, assert_inputs_exist
from scilpy.utils.filenames import split_name_with_nii

logger = logging.getLogger(__file__)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('dwi',
                        help='DWI Nifti image')
    parser.add_argument('bvals',
                        help='B-values file in FSL format')
    parser.add_argument('bvecs',
                        help='B-vectors file in FSL format')
    parser.add_argument('output',
                        help='Output b0 file(s)')
    parser.add_argument('--b0_thr', type=float, default=0.0,
                        help='All b-values with values less than or equal '
                             'to b0_thr are considered as b0s i.e. without '
                             'diffusion weighting')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true',
                       help='Extract all b0. Index number will be appended to '
                            'the output file')
    group.add_argument('--mean', action='store_true', help='Extract mean b0')

    add_verbose_arg(parser)

    return parser


def _keep_time_step(dwi, time, output):
    image = nib.load(dwi)
    data = image.get_data()

    fname, fext = split_name_with_nii(os.path.basename(output))

    multi_b0 = len(time) > 1
    for t in time:
        t_data = data[..., t]

        out_name = os.path.join(
            os.path.dirname(os.path.abspath(output)),
            '{}_{}{}'.format(fname, t, fext)) if multi_b0 else output
        nib.save(nib.Nifti1Image(t_data, image.affine, image.header),
                 out_name)


def _mean_in_time(dwi, time, output):
    image = nib.load(dwi)

    data = image.get_data()
    data = data[..., time]
    data = np.mean(data, axis=3, dtype=data.dtype)

    nib.save(nib.Nifti1Image(data, image.affine, image.header),
             output)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.dwi, args.bvals, args.bvecs])

    # We don't assert the existence of any output here because there
    # are many possible inputs/outputs.

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    bvals_min = bvals.min()

    # TODO refactor those checks
    # Should be min bval, then b0.
    if bvals_min < 0 or bvals_min > 20:
        raise ValueError(
            'The minimal b-value is lesser than 0 or greater than 20. This '
            'is highly suspicious. Please check your data to ensure '
            'everything is correct. Value found: {}'.format(bvals_min))

    b0_threshold = args.b0_thr
    if b0_threshold < 0 or b0_threshold > 20:
        raise ValueError('Invalid --b0_thr value (<0 or >20). This is highly '
                         'suspicious. Value found: {}'.format(b0_threshold))

    if not np.isclose(bvals_min, 0.0):
        b0_threshold = b0_threshold if b0_threshold > bvals_min else bvals_min
        logging.warning('No b=0 image. Setting b0_threshold to {}'.format(
                        b0_threshold))

    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
    b0_idx = np.where(gtab.b0s_mask)[0]

    logger.info('Number of b0 images in the data: {}'.format(len(b0_idx)))

    if args.mean:
        logger.info('Using mean of indices {} for b0'.format(b0_idx))
        _mean_in_time(args.dwi, b0_idx, args.output)
    else:
        if not args.all:
            b0_idx = [b0_idx[0]]
        logger.info("Keeping {} for b0".format(b0_idx))
        _keep_time_step(args.dwi, b0_idx, args.output)


if __name__ == '__main__':
    main()
