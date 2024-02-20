#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Return the number of streamlines in a tractogram. Only support trk and tck in
order to support the lazy loading from nibabel.

Formerly: scil_count_streamlines.py
"""

import argparse
import json
import logging
import os

from scilpy.io.utils import (add_json_args,
                             add_verbose_arg,
                             assert_inputs_exist)
from scilpy.tractograms.lazy_tractogram_operations import \
    lazy_streamlines_count


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file.')
    p.add_argument('--print_count_alone', action='store_true',
                   help="If true, prints the result only. \nElse, prints the "
                        "bundle name and count formatted as a json dict."
                        "(default)")

    add_json_args(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram)

    bundle_name, _ = os.path.splitext(os.path.basename(args.in_tractogram))
    count = int(lazy_streamlines_count(args.in_tractogram))

    if args.print_count_alone:
        print(count)
    else:
        stats = {
            bundle_name: {
                'streamline_count': count
            }
        }
        print(json.dumps(stats, indent=args.indent))


if __name__ == '__main__':
    main()
