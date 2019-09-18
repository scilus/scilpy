#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json

import nibabel as nib
import numpy as np

from scilpy.io.utils import add_overwrite_arg, assert_inputs_exist


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute bundle volume per label',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('voxel_label_map',
                   help='Fiber bundle file.')
    p.add_argument('bundle_name',
                   help='Bundle name.')
    p.add_argument('--indent',
                   type=int, default=2,
                   help='Indent for json pretty print. [%(default)s]')
    p.add_argument('--sort_keys',
                   action='store_true',
                   help='Sort keys in output json.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.voxel_label_map)

    voxel_label_map_img = nib.load(args.voxel_label_map)
    voxel_label_map_data = voxel_label_map_img.get_data()
    spacing = voxel_label_map_img.header['pixdim'][1:4]

    labels = np.unique(voxel_label_map_data.astype(np.uint8))[1:]
    num_digits_labels = len(str(np.max(labels)))
    voxel_volume = np.prod(spacing)
    stats = {
            args.bundle_name: {'volume': {}}
    }
    for i in labels:
        stats[args.bundle_name]['volume']['{}'.format(i)
                                              .zfill(num_digits_labels)] =\
            len(voxel_label_map_data[voxel_label_map_data == i]) * voxel_volume

    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
