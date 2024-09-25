#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop a volume using a given or an automatically computed bounding box. If a
previously computed bounding box file is given, the cropping will be applied
and the affine fixed accordingly.

Warning: This works well on masked images (like with FSL-Bet) volumes since
it's looking for non-zero data. Therefore, you should validate the results on
other types of images that haven't been masked.

Formerly: scil_crop_volume.py
"""

import argparse
import logging
import pickle

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.image.utils import compute_nifti_bounding_box
from scilpy.image.volume_operations import crop_volume


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('in_image',
                   help='Path of the nifti file to crop.')
    p.add_argument('out_image',
                   help='Path of the cropped nifti file to write.')

    p.add_argument('--ignore_voxel_size', action='store_true',
                   help='Ignore voxel size compatibility test between input '
                        'bounding box and data. Warning, use only if you '
                        'know what you are doing.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    g1 = p.add_mutually_exclusive_group()
    g1.add_argument('--input_bbox',
                    help='Path of the pickle file from which to take '
                         'the bounding box to crop input file.')
    g1.add_argument('--output_bbox',
                    help='Path of the pickle file where to write the '
                         'computed bounding box. (.pickle extension)')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image, args.input_bbox)
    assert_outputs_exist(parser, args, args.out_image, args.output_bbox)

    img = nib.load(args.in_image)
    if args.input_bbox:
        with open(args.input_bbox, 'rb') as bbox_file:
            wbbox = pickle.load(bbox_file)
        if not args.ignore_voxel_size:
            voxel_size = img.header.get_zooms()[0:3]
            if not np.allclose(voxel_size, wbbox.voxel_size[0:3], atol=1e-03):
                raise IOError("Bounding box and data voxel sizes are not "
                              "compatible. Use option --ignore_voxel_size "
                              "to ignore this test.")
    else:
        wbbox = compute_nifti_bounding_box(img)
        if args.output_bbox:
            with open(args.output_bbox, 'wb') as bbox_file:
                pickle.dump(wbbox, bbox_file)

    out_nifti_file = crop_volume(img, wbbox)
    nib.save(out_nifti_file, args.out_image)


if __name__ == "__main__":
    main()
