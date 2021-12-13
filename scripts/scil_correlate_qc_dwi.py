#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import argparse

from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.io.utils import (assert_inputs_exist,
                             add_force_b0_arg)
from scilpy.utils.bvec_bval_tools import check_b0_threshold, normalize_bvecs
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

    p.add_argument('--print_all', action='store_true',
                   help='')
    p.add_argument('--b0_thr', type=float, default=0.0,
                   help='All b-values with values less than or equal '
                        'to b0_thr are considered as b0s i.e. without '
                        'diffusion weighting. [%(default)s]')
    add_force_b0_arg(p)
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
    for bval in np.unique(bvals[bvals > b0_thr]):
        shell_idx = np.where(bvals == bval)[0]
        shell = bvecs[shell_idx]
        results_dict[bval] = np.ones((len(shell), 3)) * -1
        for i, vec in enumerate(shell):
            if np.linalg.norm(vec) < 0.001:
                continue

            dot_product = np.clip(np.tensordot(shell, vec, axes=1), -1, 1)
            # print(dot_product)
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
        avg_angle = np.average(results_dict[key][:, 1])
        std_angle = np.std(results_dict[key][:, 1])

        avg_corr = np.average(results_dict[key][:, 2])
        std_corr = np.std(results_dict[key][:, 2])

        outliers_angle = np.argwhere(
            results_dict[key][:, 1] < avg_angle-(2*std_angle))
        outliers_corr = np.argwhere(
            results_dict[key][:, 2] < avg_corr-(2*std_corr))

        print('Shell {} average and standard deviation of angle: {} +/- {}'.format(
            key, avg_angle, std_angle))
        print('Shell {} average and standard deviation of correlation: {} +/- {}'.format(
            key, avg_corr, std_corr))
        for i in outliers_angle:
            print('angle', results_dict[key][i, :])
        for i in outliers_corr:
            print('corr', results_dict[key][i, :])

        if args.print_all:
            print()
            print(results_dict)


if __name__ == "__main__":
    main()
