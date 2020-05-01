#! /usr/bin/env python
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
"""

import argparse
import json

import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_matrix',
                   help='Connectivity matrix in numpy (.npy) format.')
    p.add_argument('in_json',
                   help='Json file with the sub-network as keys and x/y '
                        'lists as value.')
    p.add_argument('out_prefix',
                   help='Prefix for the output filename.')

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

    assert_inputs_exist(parser, [args.in_matrix, args.in_json],
                        optional=args.labels_list)

    with open(args.in_json) as json_data:
        config = json.load(json_data)
    if args.keys:
        keys = args.keys
    else:
        keys = config.keys()
    out_filenames = []
    for key in keys:
        out_filenames.append("{}_{}.npy".format(args.out_prefix, key))
    assert_outputs_exist(parser, args, out_filenames)

    for i, key in enumerate(keys):
        if args.labels_list:
            labels_list = np.loadtxt(args.labels_list, dtype=np.int16).tolist()
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

        matrix = np.load(args.in_matrix)
        if (np.array(indices_1) > matrix.shape[0]).any() \
                or (indices_2 > np.array(matrix.shape[1])).any():
            raise ValueError(
                'Indices from config higher than matrix size, maybe you need a '
                'labels list?'.format(key))
        tmp_matrix = matrix[tuple(indices_1), :]
        tmp_matrix = tmp_matrix[:, tuple(indices_2)]
        np.save(out_filenames[i], tmp_matrix)


if __name__ == "__main__":
    main()
