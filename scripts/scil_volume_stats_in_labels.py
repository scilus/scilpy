#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes the information from the input map for each cortical region
(corresponding to an atlas).

Hint: For instance, this script could be useful if you have a seed map from a
specific bundle, to know from which regions it originated.

Formerly: scil_compute_seed_by_labels.py
"""

import argparse
import json
import logging

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels, get_stats_in_label
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_headers_compatible)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_labels',
                   help='Path of the input label file.')
    p.add_argument('in_labels_lut',
                   help='Path of the LUT file corresponding to labels,'
                        'used to name the regions of interest.')
    p.add_argument('in_map',
                   help='Path of the input map file. Expecting a 3D file.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser,
                        [args.in_labels, args.in_map, args.in_labels_lut])
    assert_headers_compatible(parser, [args.in_labels, args.in_map])

    # Loading
    label_data = get_data_as_labels(nib.load(args.in_labels))
    with open(args.in_labels_lut) as f:
        label_dict = json.load(f)
    map_data = nib.load(args.in_map).get_fdata(dtype=np.float32)
    if len(map_data.shape) > 3:
        parser.error('Mask should be a 3D image.')

    # Process
    out_dict = get_stats_in_label(map_data, label_data, label_dict)
    print(json.dumps(out_dict))


if __name__ == "__main__":
    main()
