#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate pair-wise similarity measures of connectivity matrix.

The computed similarity measures are:
sum of square difference and pearson correlation coefficent

Formerly: scil_evaluate_connectivity_pairwaise_agreement_measures.py
"""

import argparse
import itertools
import json
import logging

import numpy as np

from scilpy.io.utils import (add_json_args,
                             add_verbose_arg,
                             add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)
from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_matrices', nargs='+',
                   help='Path of the input matricies.')
    p.add_argument('out_json',
                   help='Path of the output json file.')
    p.add_argument('--single_compare', metavar='matrix',
                   help='Compare inputs to this single file.\n'
                        '(Else, compute all pairs in in_matrices).')
    p.add_argument('--normalize', action='store_true',
                   help='If set, will normalize all matrices from zero to '
                        'one.')

    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_matrices)
    assert_outputs_exist(parser, args, args.out_json)

    all_matrices = []
    for filename in args.in_matrices:
        tmp_mat = load_matrix_in_any_format(filename)
        tmp_mat = tmp_mat.astype(float)
        tmp_mat -= np.min(tmp_mat)
        if args.normalize:
            all_matrices.append(tmp_mat / np.max(tmp_mat))
        else:
            all_matrices.append(tmp_mat)

    if args.single_compare:
        tmp_mat = load_matrix_in_any_format(args.single_compare)
        tmp_mat = tmp_mat.astype(float)
        tmp_mat -= np.min(tmp_mat)
        if args.normalize:
            all_matrices.append(tmp_mat / np.max(tmp_mat))
        else:
            all_matrices.append(tmp_mat)

    output_measures_dict = {'RMSE': [], 'correlation': [],
                            'w_dice_voxels': [], 'dice_voxels': []}

    if args.single_compare:
        if args.single_compare in args.in_matrices:
            id = args.in_matrices.index(args.single_compare)
            all_matrices.pop(id)
        pairs = list(itertools.product(all_matrices[:-1], [all_matrices[-1]]))
    else:
        pairs = list(itertools.combinations(all_matrices, r=2))

    for i in pairs:
        rmse = np.sqrt(np.mean((i[0]-i[1])**2))
        output_measures_dict['RMSE'].append(rmse)
        corrcoef = np.corrcoef(i[0].ravel(), i[1].ravel())
        output_measures_dict['correlation'].append(corrcoef[0][1])
        dice, w_dice = compute_dice_voxel(i[0], i[1])
        output_measures_dict['dice_voxels'].append(dice)
        output_measures_dict['w_dice_voxels'].append(w_dice)

    with open(args.out_json, 'w') as outfile:
        json.dump(output_measures_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
