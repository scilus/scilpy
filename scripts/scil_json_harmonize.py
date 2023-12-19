#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script will harmonize a json file by adding missing keys and values that
differs between the different layers of the dictionary.

This is use only (for now) in Aggregate_All_* portion of tractometry-flow,
to counter the problem of missing bundles/metrics/lesions between subjects.
"""

import argparse
from copy import deepcopy
import json

from deepdiff import DeepDiff

from scilpy.io.utils import (assert_inputs_exist,
                             add_json_args,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg)
from scilpy.utils.util import recursive_update, recursive_print


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_file',
                   help='Input file (json).')
    p.add_argument('out_file',
                   help='Output file (json).')

    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_file)
    assert_outputs_exist(parser, args, args.out_file)

    with open(args.in_file) as f:
        data = json.load(f)
        data_old = deepcopy(data)

    # Generate the reference (full) dictionary. Skip level 0, complete all
    # levels, but write NaN at last level (leaf).
    new = {}
    for key in data.keys():
        new = recursive_update(new, data[key], from_existing=False)

    # Harmonize the original dictionary, missing keys are added, when a <leaf>
    # is missing NaN will be stored
    for key in data.keys():
        data[key] = recursive_update(data[key], new, from_existing=True)

    if args.verbose:
        print('Layered keys of the dictionary:')
        recursive_print(data)
        print()

        dd = DeepDiff(data, data_old, ignore_order=True)
        if 'dictionary_item_removed' in dd:
            print('Missing keys that were harmonized:')
            print(dd['dictionary_item_removed'])

    with open(args.out_file, "w") as f:
        json.dump(data, f, indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
