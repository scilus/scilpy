#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-order a connectivity matrix using a text file format.
The first row are the (x) and the second row the (y), the resulting matrix
does not have to be square (support unequal number of x and y)

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
import os

import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty)


EPILOG = """
[1] Rubinov, Mikail, and Olaf Sporns. "Complex network measures of brain
    connectivity: uses and interpretations." Neuroimage 52.3 (2010):
    1059-1069.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__, epilog=EPILOG)

    p.add_argument('in_matrix', nargs='+',
                   help='Connectivity matrix in numpy (.npy) format.')
    p.add_argument('in_ordering',
                   help='Json file with the sub-network as keys and x/y '
                        'lists as value (.txt).')
    p.add_argument('out_prefix',
                   help='Prefix for the output matrix filename.')

    p.add_argument('--out_dir',
                   help='Output directory for the Free Water results.')
    p.add_argument('--labels_list',
                   help='List saved by the decomposition script,\n'
                        'the json must contain labels rather than coordinates '
                        '(.txt).')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_matrix,
                        [args.labels_list, args.in_ordering])
    assert_output_dirs_exist_and_empty(parser, args, [], args.out_dir)
    if args.out_dir is None:
        args.out_dir = './'
    out_filenames = []
    for filename in args.in_matrix:
        basename, _ = os.path.splitext(filename)
        basename = os.path.basename(basename)
        out_filenames.append('{}/{}{}.npy'.format(args.out_dir,
                                                  args.out_prefix,
                                                  basename))

    assert_outputs_exist(parser, args, out_filenames)
    with open(args.in_ordering, 'r') as my_file:
        lines = my_file.readlines()
        ordering = [[int(val) for val in lines[0].split()],
                    [int(val) for val in lines[1].split()]]

    for filename in args.in_matrix:
        basename, _ = os.path.splitext(filename)
        basename = os.path.basename(basename)
        matrix = load_matrix_in_any_format(filename)

        if args.labels_list:
            labels_list = np.loadtxt(args.labels_list, dtype=np.int16).tolist()
            indices_1, indices_2 = [], []
            for j in ordering[0]:
                indices_1.append(labels_list.index(j))
            for j in ordering[1]:
                indices_2.append(labels_list.index(j))
        else:
            indices_1 = ordering[0]
            indices_2 = ordering[1]

        if (np.array(indices_1) > matrix.shape[0]).any() \
                or (indices_2 > np.array(matrix.shape[1])).any():
            raise ValueError('Indices from config higher than matrix size, '
                             'maybe you need a labels list?')
        tmp_matrix = matrix[tuple(indices_1), :]
        tmp_matrix = tmp_matrix[:, tuple(indices_2)]
        save_matrix_in_any_format('{}/{}{}.npy'.format(args.out_dir,
                                                       args.out_prefix,
                                                       basename),
                                  tmp_matrix)


if __name__ == "__main__":
    main()
