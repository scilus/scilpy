#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Merge multiple json file into a single one.
the --keep_separate option will add an entry for each file, the basename will
become the key.
"""

import argparse
import json
import os

from scilpy.io.utils import (add_overwrite_arg, add_json_args,
                             assert_inputs_exist, assert_outputs_exist)


def _merge_dict(dict_1, dict_2):
    new_dict = {}
    for key in dict_1.keys():
        if isinstance(dict_1[key], list):
            new_dict[key] = dict_1[key]
        else:
            new_dict[key] = [dict_1[key]]

    for key in dict_2.keys():
        if key in new_dict:
            if isinstance(dict_2[key], list):
                new_dict[key].extend(dict_2[key])
            else:
                new_dict[key].append(dict_2[key])
        else:
            if isinstance(dict_2[key], list):
                new_dict[key] = dict_2[key]
            else:
                new_dict[key] = [dict_2[key]]

    return new_dict


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_json', nargs='+',
                   help='List of json files to merge (.json).')
    p.add_argument('out_json',
                   help='Output json file (.json).')

    p.add_argument('--keep_separate', action='store_true',
                   help='Merge entries as separate keys.')

    add_json_args(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_json)
    assert_outputs_exist(parser, args, args.out_json)

    out_dict = {}
    for in_file in args.in_json:
        with open(in_file, 'r') as json_file:
            in_dict = json.load(json_file)
            if args.keep_separate:
                out_dict[os.path.splitext(in_file)[0]] = in_dict
            else:
                out_dict = _merge_dict(out_dict, in_dict)

    with open(args.out_json, 'w') as outfile:
        json.dump(out_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
