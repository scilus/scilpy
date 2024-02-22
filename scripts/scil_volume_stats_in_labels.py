#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes the information from the seeding map for each cortical region
(corresponding to an atlas) associated with a specific bundle.
Here we want to estimate the seeding attribution to cortical area
affected by the bundle

Formerly: scil_compute_seed_by_labels.py
"""

import argparse
import json
import logging

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_labels',
                   help='Path of the input label file.')
    p.add_argument('in_labels_lut',
                   help='Path of the LUT file corresponding to labels,'
                        'used to name the regions of interest.')
    p.add_argument('in_seed_maps',
                   help='Path of the input seed map file.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    required = args.in_labels, args.in_seed_maps, args.in_labels_lut
    assert_inputs_exist(parser, required)

    # Load atlas image
    label_img = nib.load(args.in_labels)
    label_img_data = get_data_as_labels(label_img)

    # Load atlas lut
    with open(args.in_labels_lut) as f:
        label_dict = json.load(f)
    (label_indices, label_names) = zip(*label_dict.items())

    # Load seed image
    seed_img = nib.load(args.in_seed_maps)
    seed_img_data = seed_img.get_fdata(dtype=np.float32)

    for label, name in zip(label_indices, label_names):
        label = int(label)
        if label != 0:
            curr_data = (seed_img_data[np.where(label_img_data == label)])
            nb_vx_roi = np.count_nonzero(label_img_data == label)
            nb_seed_vx = np.count_nonzero(curr_data)

            if nb_seed_vx != 0:
                mean_seed = np.sum(curr_data)/nb_seed_vx
                max_seed = np.max(curr_data)
                std_seed = np.sqrt(np.mean(abs(curr_data[curr_data != 0] -
                                               mean_seed)**2))

                print(json.dumps({'ROI-idx': label,
                                  'ROI-name': str(name),
                                  'nb-vx-roi': int(nb_vx_roi),
                                  'nb-vx-seed': int(nb_seed_vx),
                                  'max': int(max_seed),
                                  'mean': float(mean_seed),
                                  'std': float(std_seed)}))


if __name__ == "__main__":
    main()
