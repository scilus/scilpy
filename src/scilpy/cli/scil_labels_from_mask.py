#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to obtain labels from a binary mask which contains multiple blobs.

The script will assign a label to each blob in the mask. The background label
is excluded from the labels list. By default, the background label is 0 and
the labels are assigned in increasing order starting from 1.
"""


import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_labels_from_mask
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             add_verbose_arg, assert_outputs_exist)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_mask', type=str, help='Input mask file.')
    p.add_argument('out_labels', type=str, help='Output label file.')

    p.add_argument('--labels', nargs='+', default=None, type=int,
                   help='Labels to assign to each blobs in the mask. '
                        'Excludes the background label.')
    p.add_argument('--background_label', default=0, type=int,
                   help='Label to assign to the background. [%(default)s]')

    p.add_argument('--min_volume', type=float, default=7,
                   help='Minimum volume in mm3 [%(default)s],'
                        'Useful for lesions.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_mask)
    assert_outputs_exist(parser, args, args.out_labels)
    # Load mask and get data
    mask_img = nib.load(args.in_mask)
    mask_data = get_data_as_mask(mask_img)
    voxel_volume = np.prod(np.diag(mask_img.affine)[:3])
    min_voxel_count = args.min_volume // voxel_volume

    # Get labels from mask
    label_map = get_labels_from_mask(
        mask_data, args.labels, args.background_label,
        min_voxel_count=min_voxel_count)
    # Save result
    out_img = nib.Nifti1Image(label_map.astype(np.uint16), mask_img.affine)
    nib.save(out_img, args.out_labels)


if __name__ == "__main__":
    main()
