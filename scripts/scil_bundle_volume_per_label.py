#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute bundle volume per label in mm³. This script supports anisotropic voxels
resolution. Volume is estimated by counting the number of voxel occupied by
each label and multiplying it by the volume of a single voxel.

This estimation is typically performed at resolution around 1mm³.
"""

import argparse
import json

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             assert_inputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('voxel_label_map',
                   help='Fiber bundle file.')
    p.add_argument('bundle_name',
                   help='Bundle name.')

    add_json_args(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.voxel_label_map)

    voxel_label_map_img = nib.load(args.voxel_label_map)
    voxel_label_map_data = voxel_label_map_img.get_data()
    voxel_size = voxel_label_map_img.header['pixdim'][1:4]

    labels = np.unique(voxel_label_map_data.astype(np.uint8))[1:]
    num_digits_labels = len(str(np.max(labels)))
    voxel_volume = np.prod(voxel_size)
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
