#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Merge multiple json file into a single one.
the --keep_separate option will add an entry for each file, the basename will
become the key.
"""

import argparse
import json
import os

import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_json_args,
                             assert_inputs_exist, assert_outputs_exist)


def _merge_dict(dict_1, dict_2, no_list=False, recursive=False):
    new_dict = {}
    for key in dict_1.keys():
        new_dict[key] = dict_1[key]

    for key in dict_2.keys():
        if isinstance(dict_2[key], dict) and recursive:
            if key not in dict_1:
                dict_1[key] = {}
            new_dict[key] = _merge_dict(dict_1[key], dict_2[key],
                                        no_list=no_list, recursive=recursive)
        elif key not in new_dict:
            new_dict[key] = dict_2[key]
        else:
            if not isinstance(new_dict[key], list) and not no_list:
                new_dict[key] = [new_dict[key]]

            if not isinstance(dict_2[key], list) and not no_list:
                new_dict[key].extend([dict_2[key]])
            else:
                if isinstance(dict_2[key], dict):
                    new_dict.update(dict_2)
                else:
                    new_dict[key] = new_dict[key] + dict_2[key]

    return new_dict


def _average_dict(dict_1):
    for key in dict_1.keys():
        if isinstance(dict_1[key], dict):
            dict_1[key] = _average_dict(dict_1[key])
        elif isinstance(dict_1[key], list) or np.isscalar(dict_1[key]):
            new_dict = {}
            for subkey in dict_1.keys():
                new_dict[subkey] = {'mean': np.average(dict_1[subkey]),
                                    'std': np.std(dict_1[subkey])}
            return new_dict

    return dict_1


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_json', nargs='+',
                   help='List of json files to merge (.json).')
    p.add_argument('out_json',
                   help='Output json file (.json).')

    p.add_argument('--keep_separate', action='store_true',
                   help='Merge entries as separate keys based on filename.')
    p.add_argument('--no_list', action='store_true',
                   help='Merge entries knowing there is no conflict.')
    p.add_argument('--add_parent_key',
                   help='Merge all entries under a single parent.')
    p.add_argument('--remove_parent_key', action='store_true',
                   help='Merge ignoring parent key (e.g for population).')
    p.add_argument('--recursive', action='store_true',
                   help='Merge all entries at the lowest layers.')
    p.add_argument('--average_last_layer', action='store_true',
                   help='Average all entries at the lowest layers.')
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
            if args.remove_parent_key:
                in_dict = list(in_dict.values())[0]
            if args.keep_separate:
                out_dict[os.path.splitext(in_file)[0]] = in_dict
            else:
                out_dict = _merge_dict(out_dict, in_dict,
                                       no_list=args.no_list,
                                       recursive=args.recursive)

    if args.average_last_layer:
        out_dict = _average_dict(out_dict)

    with open(args.out_json, 'w') as outfile:
        if args.add_parent_key:
            out_dict = {args.add_parent_key: out_dict}
        json.dump(out_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
