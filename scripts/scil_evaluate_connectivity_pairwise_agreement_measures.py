#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate pair-wise similarity measures of bundles.
All tractograms must be in the same space (aligned to one reference)

For the voxel representation the computed similarity measures are:
bundle_adjacency_voxels, dice_voxels, w_dice_voxels, density_correlation
volume_overlap, volume_overreach
The same measures are also evluated for the endpoints.

For the streamline representation the computed similarity measures are:
bundle_adjacency_streamlines, dice_streamlines, streamlines_count_overlap,
streamlines_count_overreach
"""

import argparse
import itertools
import json
import logging
import os
import shutil

import numpy as np

from scilpy.io.utils import (add_json_args,
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
                   help='Path of the input bundles.')
    p.add_argument('out_json',
                   help='Path of the output json file.')
    p.add_argument('--single_compare',
                   help='Compare inputs to this single file.')

    add_json_args(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_matrices)
    assert_outputs_exist(parser, args, [args.out_json])

    # The input can be 2, 10 or 100+ matrices
    all_matrices = []
    for filename in args.in_matrices:
        tmp_mat = load_matrix_in_any_format(filename)
        all_matrices.append(tmp_mat / np.max(tmp_mat))

    # SSD of first two as an example
    ssd = np.sum((all_matrices[0] - all_matrices[1])**2)
    output_measures_dict = {'SSD': [ssd]}

    # But for the real script we need ALL the pairs:
    # 2 input means one pair, 3 input means 2 pairs (1-2, 1-3, 2-3)
    # N input means Nx(N-1) / 2 pairs

    # Single compare means you compare everyone against a single one.
    # All that pairing is in the scil_evaluate_bundles_pairwise_agreement_measures.py

    # The output is a dictionnary with keys being the metrics (SSD, Correlation)
    # The entries are list of results on length NUMBER_OF_PAIRS
    with open(args.out_json, 'w') as outfile:
        json.dump(output_measures_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
