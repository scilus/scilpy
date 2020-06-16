#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes the information from the seeding map for each cortical region
(corresponding to an atlas) associated with a specific bundle.
Here we want to estimate the seeding attribution to cortical area
affected by the bundle
"""

import argparse
import json

import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_label
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('labels',
                   help='Path of the input atlas file.')
    p.add_argument('labels_lut',
                   help='Path of the lookup table file, '
                        'used to name the regions of interest.')
    p.add_argument('seed_maps',
                   help='Path of the input seeding file')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.labels, args.seed_maps)

    # Load atlas image
    label_img = nib.load(args.labels)
    label_img_data = get_data_as_label(label_img)

    # Load atlas lut
    with open(args.labels_lut) as f:
        label_dict = json.load(f)
    (label_indices, label_names) = zip(*label_dict.items())

    # Load seed image
    seed_img = nib.load(args.seed_maps)
    seed_img_data = seed_img.get_fdata(dtype=np.float32)

    for label, name in zip(label_indices, label_names):
        if int(label) != 0:

            curr_data = (seed_img_data[np.where(label_img_data == int(label))])
            nb_vx_roi = np.count_nonzero(label_img_data == int(label))
            nb_seed_vx = np.count_nonzero(curr_data)

            # fix issue from the presence of NaN or absence voxel seed in ROI
            np.seterr(divide='ignore', invalid='ignore')

            mean_seed = np.sum(curr_data)/nb_seed_vx
            max_seed = np.max(curr_data)
            std_seed = np.sqrt(np.mean(abs(curr_data[curr_data != 0] - mean_seed)**2))
            if nb_seed_vx != 0:
                print(json.dumps({'ROI-idx': label, 'ROI-name': str(name), 'nb-vx-roi': int(nb_vx_roi), 'nb-vx-seed': int(nb_seed_vx), 'max': int(max_seed), 'mean': float(mean_seed), 'std': float(std_seed)}))


if __name__ == "__main__":
    main()
