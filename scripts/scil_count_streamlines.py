#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import os

import nibabel as nib

from scilpy.io.utils import assert_inputs_exist


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Returns the number of streamlines in a bundle',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('bundle', help='Fiber bundle file')
    parser.add_argument('--indent', type=int, default=2,
                        help='Indent for json pretty print')
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.bundle])

    bundle_name, _ = os.path.splitext(os.path.basename(args.bundle))
    bundle_tractogram_file = nib.streamlines.load(args.bundle, lazy_load=True)
    stats = {
        bundle_name: {
            'tract_count': int(bundle_tractogram_file.header['nb_streamlines'])
        }
    }

    print(json.dumps(stats, indent=args.indent))


if __name__ == '__main__':
    main()
