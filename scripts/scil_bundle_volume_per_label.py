#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute bundle volume per label in mm3. This script supports anisotropic voxels
resolution. Volume is estimated by counting the number of voxel occupied by
each label and multiplying it by the volume of a single voxel.

The labels can be obtained by scil_bundle_label_map.py.

This estimation is typically performed at resolution around 1mm3.

To get the volume and other measures directly from the (whole) bundle, use
scil_bundle_shape_measures.py.

Formerly: scil_compute_bundle_volume_per_label.py
"""

import argparse
import json
import logging

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.io.utils import (add_json_args,
                             add_verbose_arg,
                             add_overwrite_arg,
                             assert_inputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('voxel_label_map',
                   help='Fiber bundle file.')
    p.add_argument('bundle_name',
                   help='Bundle name.')

    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.voxel_label_map)

    voxel_label_map_img = nib.load(args.voxel_label_map)
    voxel_label_map_data = get_data_as_labels(voxel_label_map_img)
    voxel_size = voxel_label_map_img.header['pixdim'][1:4]

    labels = np.unique(voxel_label_map_data.astype(np.uint8))[1:]
    num_digits_labels = 3
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
