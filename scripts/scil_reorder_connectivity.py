#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-order a connectivity matrix using a json file in a format such as:
    {"temporal": [[1,3,5,7], [0,2,4,6]]}.
The key is to identify the sub-network, the first list is for the
column (x) and the second is for the row (y).

The values refers to the coordinates (starting at 0) in the matrix, but if the
--labels_list parameter is used, the values will refers to the label which will
be converted to the appropriate coordinates. This file must be the same as the
one provided to the scil_decompose_connectivity.py

To subsequently use scil_visualize_connectivity.py with a lookup table, you
must use a label-based reording json and use --labels_list.

The option bct_reorder_nodes creates its own ordering scheme that will be saved
and then applied to others.
We recommand running this option on a population-averaged matrix.
The results are stochastic due to simulated annealing.

This script is under the GNU GPLv3 license, for more detail please refer to
https://www.gnu.org/licenses/gpl-3.0.en.html
"""

import argparse
import json

import bct
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format,
                             assert_outputs_exist)


EPILOG = """
[1] Rubinov, Mikail, and Olaf Sporns. "Complex network measures of brain
    connectivity: uses and interpretations." Neuroimage 52.3 (2010):
    1059-1069.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__, epilog=EPILOG)

    p.add_argument('in_matrix',
                   help='Connectivity matrix in numpy (.npy) format.')
    p.add_argument('out_prefix',
                   help='Prefix for the output matrix filename.')

    reorder = p.add_mutually_exclusive_group(required=True)
    reorder.add_argument('--in_json',
                         help='Json file with the sub-network as keys and x/y '
                              'lists as value.')
    reorder.add_argument('--bct_reorder_nodes', metavar='OUT_JSON',
                         help='Rearranges the nodes so the elements are '
                              'squeezed along the main diagonal.')

    p.add_argument('--keys', nargs='+',
                   help='Only generate the specified sub-network.')
    p.add_argument('--labels_list',
                   help='List saved by the decomposition script,\n'
                        'the json must contain labels rather than coordinates.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_matrix,
                        [args.labels_list, args.in_json])

    if args.labels_list and args.bct_reorder_nodes:
        parser.error('Cannot use the bct_reorder_nodes option with a '
                     'labels_list.')

    if args.keys and args.bct_reorder_nodes:
        parser.error('Cannot use the bct_reorder_nodes option with keys.')

    matrix = load_matrix_in_any_format(args.in_matrix)

    if args.in_json:
        with open(args.in_json) as json_data:
            config = json.load(json_data)
        if args.keys:
            keys = args.keys
        else:
            keys = config.keys()
        out_filenames = []
        for key in keys:
            out_filenames.append('{}_{}.npy'.format(args.out_prefix, key))
        assert_outputs_exist(parser, args, out_filenames)

        for i, key in enumerate(keys):
            if args.labels_list:
                labels_list = np.loadtxt(
                    args.labels_list, dtype=np.int16).tolist()
                indices_1, indices_2 = [], []
                for j in config[key][0]:
                    indices_1.append(labels_list.index(j))
                for j in config[key][1]:
                    indices_2.append(labels_list.index(j))
            else:
                if key not in config:
                    raise ValueError('{} not in config, maybe you need a labels '
                                     'list?'.format(key))
                indices_1 = config[key][0]
                indices_2 = config[key][1]

            if (np.array(indices_1) > matrix.shape[0]).any() \
                    or (indices_2 > np.array(matrix.shape[1])).any():
                raise ValueError('Indices from config higher than matrix size, '
                                 'maybe you need a labels list?')
            tmp_matrix = matrix[tuple(indices_1), :]
            tmp_matrix = tmp_matrix[:, tuple(indices_2)]
            save_matrix_in_any_format(out_filenames[i], tmp_matrix)
    else:
        assert_outputs_exist(parser, args, [], args.bct_reorder_nodes)

        out_matrix, out_indices, _ = bct.reorder_matrix(matrix)
        out_json = args.bct_reorder_nodes

        out_indices = out_indices.tolist()
        save_matrix_in_any_format('{}_{}.npy'.format(
            args.out_prefix, 'bct_reorder_nodes'), out_matrix)
        out_indices_dict = {'bct_reorder_nodes': [out_indices,
                                                  out_indices]}

        with open(out_json, 'w') as outfile:
            json.dump(out_indices_dict, outfile, indent=2)


if __name__ == "__main__":
    main()
