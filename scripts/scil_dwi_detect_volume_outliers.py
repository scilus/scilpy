#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script simply finds the 3 closest angular neighbors of each direction
(per shell) and compute the voxel-wise correlation.
If the angles or correlations to neighbors are below the shell average (by
args.std_scale x STD) it will flag the volume as a potential outlier.

This script supports multi-shells, but each shell is independant and detected
using the --b0_threshold parameter.

This script can be run before any processing to identify potential problem
before launching pre-processing.
"""

import argparse
import logging

from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib


from scilpy.dwi.operations import detect_volume_outliers
from scilpy.io.utils import (add_b0_thresh_arg, add_skip_b0_check_arg,
                             add_verbose_arg, assert_inputs_exist, )
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

    p.add_argument('--std_scale', type=float, default=2.0,
                   help='How many deviation from the mean are required to be '
                        'considered an outlier. [%(default)s]')

    add_b0_thresh_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=True)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose == "WARNING":
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    data = nib.load(args.in_dwi).get_fdata()

    args.b0_threshold = check_b0_threshold(bvals.min(),
                                           b0_thr=args.b0_threshold,
                                           skip_b0_check=args.skip_b0_check)
    bvecs = normalize_bvecs(bvecs)

    # Not using the result. Only printing on screen. This is why the logging
    # level can never be set higher than INFO.
    detect_volume_outliers(data, bvals, bvecs, args.std_scale,
                           args.b0_threshold)


if __name__ == "__main__":
    main()
