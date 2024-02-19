#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script simply finds the 3 closest angular neighbors of each direction
(per shell) and compute the voxel-wise correlation.
If the angles or correlations to neighbors are below the shell average (by
args.std_scale x STD) it will flag the volume as a potential outlier.

This script supports multi-shells, but each shell is independant and detected
using the args.b0_thr parameter.

This script can be run before any processing to identify potential problem
before launching pre-processing.
"""

import argparse
import logging
import pprint

from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib

from scilpy.dwi.operations import detect_volume_outliers
from scilpy.io.utils import (assert_inputs_exist,
                             add_force_b0_arg,
                             add_verbose_arg)
from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              normalize_bvecs)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dwi',
                   help='The DWI file (.nii) to concatenate.')
    p.add_argument('in_bval',
                   help='The b-values files in FSL format (.bval).')
    p.add_argument('in_bvec',
                   help='The b-vectors files in FSL format (.bvec).')

    p.add_argument('--b0_thr', type=float, default=20.0,
                   help='All b-values with values less than or equal '
                        'to b0_thr are considered as b0s i.e. without '
                        'diffusion weighting. [%(default)s]')
    p.add_argument('--std_scale', type=float, default=2.0,
                   help='How many deviation from the mean are required to be '
                        'considered an outlier. [%(default)s]')

    add_force_b0_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    data = nib.load(args.in_dwi).get_fdata()

    b0_thr = check_b0_threshold(args.force_b0_threshold,
                                bvals.min(), args.b0_thr)
    bvecs = normalize_bvecs(bvecs)

    detect_volume_outliers(data, bvecs, bvals, args.std_scale,
                           args.verbose, b0_thr)


if __name__ == "__main__":
    main()
