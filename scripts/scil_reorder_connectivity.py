#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description='Flip one or more axes of the '
                                            'encoding scheme matrix.')

    p.add_argument('in_matrix',
                   help='')
    p.add_argument('in_json',
                   help='')
    p.add_argument('out_prefix',
                   help='')

    p.add_argument('--keys', nargs='+',
                   help='')
    p.add_argument('--labels_list',
                   help='')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # assert_inputs_exist(parser, args.encoding_file)
    # assert_outputs_exist(parser, args, args.flipped_encoding)

    with open(args.in_json) as json_data:
        config = json.load(json_data)

    if args.keys:
        keys = args.keys
    else:
        keys = config.keys()

    for key in keys:
        if args.labels_list:
            labels_list = np.loadtxt(args.labels_list).astype(np.int16)
            indices_1 = labels_list[config[key][0]]
            indices_2 = labels_list[config[key][1]]
        else:
            indices_1 = config[key][0]
            indices_2 = config[key][1]

        matrix = np.load(args.in_matrix)
        tmp_matrix = matrix[tuple(indices_1), :]
        tmp_matrix = tmp_matrix[:, tuple(indices_2)]
        np.save("{}_{}".format(args.out_prefix, key), tmp_matrix)


if __name__ == "__main__":
    main()
