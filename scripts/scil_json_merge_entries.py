#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Merge multiple json files into a single one.
Typically used during the tractometry pipeline.

Without option it will simply merge all entries at the top level, the top
level must not have any conflicting keys.

--keep_separate option will add a parent for each file, its basename will
become the key.

--no_list option will merge all entries at the top level, if there is a
conflict the lowest level will be extended with the new values (if list) or
added (if value)

--add_parent_key option will add a parent key before merging all entries.

--remove_parent_key option will remove the parent key before merging all
entries.

--recursive option will merge all entries (scalar) at the lowest layers as a
list.

--average_last_layer option will average all entries (scalar) at the lowest
layers, but instead of creating a list it creates a mean/std level.

Formerly: scil_merge_json.py
"""

import argparse
import json
import logging
import os

from scilpy.io.utils import (add_overwrite_arg, add_json_args,
                             assert_inputs_exist, add_verbose_arg,
                             assert_outputs_exist)
from scilpy.tractanalysis.json_utils import merge_dict, average_dict


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
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

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
                out_dict = merge_dict(out_dict, in_dict,
                                      no_list=args.no_list,
                                      recursive=args.recursive)

    if args.average_last_layer:
        out_dict = average_dict(out_dict)

    with open(args.out_json, 'w') as outfile:
        if args.add_parent_key:
            out_dict = {args.add_parent_key: out_dict}
        json.dump(out_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
