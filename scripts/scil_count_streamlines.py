#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import os

import nibabel as nib

from scilpy.io.utils import add_json_arg, assert_inputs_exist


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Return the number of streamlines in a tractogram',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('tractogram',
                        metavar='TRACTOGRAM',
                        help='path of the tracts file, in a format supported' +
                        ' by nibabel')
    add_json_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.tractogram)

    bundle_name, _ = os.path.splitext(os.path.basename(args.tractogram))
    bundle_tractogram_file = nib.streamlines.load(args.tractogram,
                                                  lazy_load=True)
    stats = {
        bundle_name: {
            'tract_count': int(bundle_tractogram_file.header['nb_streamlines'])
        }
    }

    print(json.dumps(stats, args.indent))


if __name__ == '__main__':
    main()
