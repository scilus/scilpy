#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script harmonizes labels across a set of lesion files represented in
NIfTI format. It ensures that labels are consistent across multiple input
images by matching labels between images based on spatial proximity and
overlap criteria.

The script works iteratively, so the multiple inputs should be in chronological
order (and changing the order affects the output). All images should be
co-registered.

To obtain labels from binary mask use scil_labels_from_mask.

WARNING: this script requires all files to have all lesions segmented.
If your data only show new lesions at each timepoints (common in manual
segmentation), use the option --incremental_lesions to merge past timepoints.
    T1 = T1, T2 = T1 + T2, T3 = T1 + T2 + T3
"""

import argparse
import os

import nibabel as nib
import numpy as np

from scilpy.image.labels import (get_data_as_labels, harmonize_labels,
                                 get_labels_from_mask)
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             assert_headers_compatible)

EPILOG = """
Reference:
    [1] Köhler, Caroline, et al. "Exploring individual multiple sclerosis
    lesion volume change over time: development of an algorithm for the
    analyses of longitudinal quantitative MRI measures."
    NeuroImage: Clinical 21 (2019): 101623.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_images', nargs='+',
                   help='Input file name, in nifti format.')
    p.add_argument('out_dir',
                   help='Output directory.')
    p.add_argument('--max_adjacency', type=float, default=5.0,
                   help='Maximum adjacency distance between lesions for '
                   'them to be considered as the potential match '
                   '[%(default)s].')
    p.add_argument('--min_voxel_overlap', type=int, default=1,
                   help='Minimum number of overlapping voxels between '
                   'lesions for them to be considered as the potential '
                   'match [%(default)s].')

    p.add_argument('--incremental_lesions', action='store_true',
                   help='If lesions files only show new lesions at each '
                        'timepoint, this will merge past timepoints.')
    p.add_argument('--debug_mode', action='store_true',
                   help='Add a fake voxel to the corner to ensure consistent '
                        'colors in MI-Brain.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_images)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)
    assert_headers_compatible(parser, args.in_images)

    imgs = [nib.load(filename) for filename in args.in_images]
    original_data = [get_data_as_labels(img) for img in imgs]

    masks = []
    if args.incremental_lesions:
        for i, data in enumerate(original_data):
            mask = np.zeros_like(data)
            mask[data > 0] = 1
            masks.append(mask)
            if i > 0:
                new_data = np.sum(masks, axis=0)
                new_data[new_data > 0] = 1
            else:
                new_data = mask
            original_data[i] = get_labels_from_mask(new_data)

    relabeled_data = harmonize_labels(original_data,
                                      args.min_voxel_overlap,
                                      max_adjacency=args.max_adjacency)

    max_label = np.max(relabeled_data) + 1
    for i, img in enumerate(imgs):
        if args.debug_mode:
            relabeled_data[i][0, 0, 0] = max_label  # To force identical color
        nib.save(nib.Nifti1Image(relabeled_data[i], img.affine),
                 os.path.join(args.out_dir, os.path.basename(args.in_images[i])))


if __name__ == "__main__":
    main()
