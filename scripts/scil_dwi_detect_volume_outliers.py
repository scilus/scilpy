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
import pprint

from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.io.utils import (assert_inputs_exist,
                             add_force_b0_arg,
                             add_verbose_arg)
from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              identify_shells,
                                              normalize_bvecs,
                                              round_bvals_to_shell)
import math


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
                        'considered an outliers. [%(default)s]')

    add_force_b0_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    dwi = nib.load(args.in_dwi)
    data = dwi.get_fdata()

    b0_thr = check_b0_threshold(args.force_b0_threshold,
                                bvals.min(), args.b0_thr)
    bvecs = normalize_bvecs(bvecs)

    results_dict = {}
    shells_to_extract = identify_shells(bvals, b0_thr, sort=True)[0]
    bvals = round_bvals_to_shell(bvals, shells_to_extract)
    for bval in shells_to_extract[shells_to_extract > args.b0_thr]:
        shell_idx = np.where(bvals == bval)[0]
        shell = bvecs[shell_idx]
        results_dict[bval] = np.ones((len(shell), 3)) * -1
        for i, vec in enumerate(shell):
            if np.linalg.norm(vec) < 0.001:
                continue

            dot_product = np.clip(np.tensordot(shell, vec, axes=1), -1, 1)
            angle = np.arccos(dot_product) * 180 / math.pi
            angle[np.isnan(angle)] = 0
            idx = np.argpartition(angle, 4).tolist()
            idx.remove(i)

            avg_angle = np.average(angle[idx[:3]])
            corr = np.corrcoef([data[..., shell_idx[i]].ravel(),
                                data[..., shell_idx[idx[0]]].ravel(),
                                data[..., shell_idx[idx[1]]].ravel(),
                                data[..., shell_idx[idx[2]]].ravel()])
            results_dict[bval][i] = [shell_idx[i], avg_angle,
                                     np.average(corr[0, 1:])]

    for key in results_dict.keys():
        avg_angle = np.round(np.average(results_dict[key][:, 1]), 4)
        std_angle = np.round(np.std(results_dict[key][:, 1]), 4)

        avg_corr = np.round(np.average(results_dict[key][:, 2]), 4)
        std_corr = np.round(np.std(results_dict[key][:, 2]), 4)

        outliers_angle = np.argwhere(
            results_dict[key][:, 1] < avg_angle-(args.std_scale*std_angle))
        outliers_corr = np.argwhere(
            results_dict[key][:, 2] < avg_corr-(args.std_scale*std_corr))

        print('Results for shell {} with {} directions:'.format(
            key, len(results_dict[key])))
        print('AVG and STD of angles: {} +/- {}'.format(
            avg_angle, std_angle))
        print('AVG and STD of correlations: {} +/- {}'.format(
            avg_corr, std_corr))

        if len(outliers_angle) or len(outliers_corr):
            print('Possible outliers ({} STD below or above average):'.format(
                args.std_scale))
            print('Outliers based on angle [position (4D), value]')
            for i in outliers_angle:
                print(results_dict[key][i, :][0][0:2])
            print('Outliers based on correlation [position (4D), value]')
            for i in outliers_corr:
                print(results_dict[key][i, :][0][0::2])
        else:
            print('No outliers detected.')

        if args.verbose:
            print('Shell with b-value {}'.format(key))
            pprint.pprint(results_dict[key])
        print()


if __name__ == "__main__":
    main()
