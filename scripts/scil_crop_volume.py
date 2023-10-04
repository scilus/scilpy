#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop a volume using a given or an automatically computed bounding box. If a
previously computed bounding box file is given, the cropping will be applied
and the affine fixed accordingly.

Warning: This works well on masked images (like with FSL-Bet) volumes since
it's looking for non-zero data. Therefore, you should validate the results on
other types of images that haven't been masked.
"""

import argparse
import pickle

from dipy.segment.mask import crop, bounding_box
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.util import voxel_to_world, world_to_voxel

# TODO move that elsewhere


class WorldBoundingBox(object):
    def __init__(self, minimums, maximums, voxel_size):
        self.minimums = minimums
        self.maximums = maximums
        self.voxel_size = voxel_size


def compute_nifti_bounding_box(img):
    """Finds bounding box from data and transforms it in world space for use
    on data with different attributes like voxel size."""
    data = img.get_fdata(dtype=np.float32, caching='unchanged')
    affine = img.affine
    voxel_size = img.header.get_zooms()[0:3]

    voxel_bb_mins, voxel_bb_maxs = bounding_box(data)

    world_bb_mins = voxel_to_world(voxel_bb_mins, affine)
    world_bb_maxs = voxel_to_world(voxel_bb_maxs, affine)
    wbbox = WorldBoundingBox(world_bb_mins, world_bb_maxs, voxel_size)

    return wbbox


def crop_nifti(img, wbbox):
    """Applies cropping from a world space defined bounding box and fixes the
    affine to keep data aligned."""
    data = img.get_fdata(dtype=np.float32, caching='unchanged')
    affine = img.affine

    voxel_bb_mins = world_to_voxel(wbbox.minimums, affine)
    voxel_bb_maxs = world_to_voxel(wbbox.maximums, affine)

    # Prevent from trying to crop outside data boundaries by clipping bbox
    extent = list(data.shape[:3])
    for i in range(3):
        voxel_bb_mins[i] = max(0, voxel_bb_mins[i])
        voxel_bb_maxs[i] = min(extent[i], voxel_bb_maxs[i])
    translation = voxel_to_world(voxel_bb_mins, affine)

    data_crop = np.copy(crop(data, voxel_bb_mins, voxel_bb_maxs))

    new_affine = np.copy(affine)
    new_affine[0:3, 3] = translation[0:3]

    return nib.Nifti1Image(data_crop, new_affine)


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
    add_overwrite_arg(p)

    g1 = p.add_mutually_exclusive_group()
    g1.add_argument('--input_bbox',
                    help='Path of the pickle file from which to take '
                         'the bounding box to crop input file.')
    g1.add_argument('--output_bbox',
                    help='Path of the pickle file where to write the '
                         'computed bounding box.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

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

    out_nifti_file = crop_nifti(img, wbbox)
    nib.save(out_nifti_file, args.out_image)


if __name__ == "__main__":
    main()
