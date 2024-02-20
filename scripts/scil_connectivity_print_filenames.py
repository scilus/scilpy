#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Output the list of filenames using the coordinates from a binary connectivity
matrix. Typically used to move around files that are considered valid after
the scil_connectivity_filter.py script.

Example:
# Keep connections with more than 1000 streamlines for 100% of a population
scil_connectivity_filter.py filtering_mask.npy
    --greater_than */streamlines_count.npy 1000 1.0
scil_connectivity_print_filenames.py filtering_mask.npy
    labels_list.txt pass.txt
for file in $(cat pass.txt);
    do mv ${SOMEWHERE}/${FILE} ${SOMEWHERE_ELSE}/;
done

Formerly: scil_print_connectivity_filenames.py
"""

import argparse
import logging

import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_matrix',
                   help='Binary matrix in numpy (.npy) format.\n'
                        'Typically from scil_connectivity_filter.py')
    p.add_argument('labels_list',
                   help='List saved by the decomposition script.')
    p.add_argument('out_txt',
                   help='Output text file containing all filenames.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_matrix)
    assert_outputs_exist(parser, args, args.out_txt)

    matrix = load_matrix_in_any_format(args.in_matrix)
    labels_list = np.loadtxt(args.labels_list).astype(np.uint16)

    text_file = open(args.out_txt, 'w')
    for pos_1, pos_2 in np.argwhere(matrix > 0):
        in_label = labels_list[pos_1]
        out_label = labels_list[pos_2]

        # scil_tractogram_segment_bundles_for_connectivity.py only save the
        # lower triangular files
        if out_label < in_label:
            continue
        text_file.write('{}_{}.trk\n'.format(in_label, out_label))
    text_file.close()


if __name__ == "__main__":
    main()
