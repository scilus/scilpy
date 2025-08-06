#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate pair-wise similarity measures of connectivity matrices.

The computed similarity measures are:
- RMSE: root-mean-square difference
- Pearson correlation coefficent
- w-dice: weighted dice, agreement of the values.
- dice: agreement of the binarized matrices

If more than two matrices are given in input, the similarity measures will be
computed for each pair. Alternatively, you can compare all matrices to a
single reference, using --single_compare.

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
from scilpy.stats.matrix_stats import pairwise_agreement
from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)
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

    assert_inputs_exist(parser, args.in_matrices, args.single_compare)
    assert_outputs_exist(parser, args, args.out_json)

    # Check that --single_compare is not loaded twice
    if args.single_compare and args.single_compare in args.in_matrices:
        id = args.in_matrices.index(args.single_compare)
        args.in_matrices.pop(id)

    # Load matrices
    all_matrices = []
    for filename in args.in_matrices:
        all_matrices.append(load_matrix_in_any_format(filename).astype(float))

    ref_matrix = None
    if args.single_compare:
        ref_matrix = load_matrix_in_any_format(
            args.single_compare).astype(float)

    # Compute and save
    output_measures_dict = pairwise_agreement(all_matrices, ref_matrix,
                                              normalize=args.normalize)
    with open(args.out_json, 'w') as outfile:
        json.dump(output_measures_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
