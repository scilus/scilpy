#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Return the number of streamlines in a tractogram. Only support trk and tck in
order to support the lazy loading from nibabel.
"""

import argparse
import json
import os

import nibabel as nib

from scilpy.io.utils import add_json_args, assert_inputs_exist


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file.')
    add_json_args(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)

    bundle_name, _ = os.path.splitext(os.path.basename(args.in_tractogram))
    bundle_tractogram_file = nib.streamlines.load(args.in_tractogram,
                                                  lazy_load=True)

    _, ext = os.path.splitext(args.in_tractogram)
    if ext == '.trk':
        key = 'nb_streamlines'
    elif ext == '.tck':
        key = 'count'
    else:
        parser.error('{} is not a supported extension for lazy loading'.format(ext))

    stats = {
        bundle_name: {
            'tract_count': int(bundle_tractogram_file.header[key])
        }
    }

    print(json.dumps(stats, indent=args.indent))


if __name__ == '__main__':
    main()
