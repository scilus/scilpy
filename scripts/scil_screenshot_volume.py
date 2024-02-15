#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Take screenshot(s) of one or more slices in a given image volume along the
requested axis. If slice indices are not provided, all slices in the volume 
are used. The name of the outputed images are suffixed with _slice_{id}, with 
id being the slice number in the volume. If a labelmap image is provided (e.g. 
a tissue segmentation map), it is overlaid on the volume slices. Same goes if 
a mask is provided, with the difference that it can be rendered as a 
transparency overlay as well as a contour. 

A labelmap image can be provided as the image volume, without requiring it as
the optional argument if only the former needs to be plot.

Example:
python scil_screenshot_volume.py \
  t1.nii.gz \
  t1_axial.png

python scil_screenshot_volume.py \
  t1.nii.gz \
  t1_axial.png \
  30 40 50 60 70 80 90 100

python scil_screenshot_volume.py \
  t1.nii.gz \
  t1_sagittal.png \
  --axis_name sagittal

python scil_screenshot_volume.py \
  t1.nii.gz \
  t1_axial_plasma_cmap.png \
  30 40 50 60 70 80 90 100 \
  --vol_cmap_name plasma

python scil_screenshot_volume.py \
  t1.nii.gz \
  t1_mask_overlay.png \
  30 40 50 60 70 80 90 100
  --in_mask brain_mask.nii.gz

python scil_screenshot_volume.py \
  t1.nii.gz \
  t1_mask_contour.png \
  30 40 50 60 70 80 90 100
  --in_mask brain_mask.nii.gz
  --mask_as_contour

python scil_screenshot_volume.py \
  t1.nii.gz \
  t1_axial_tissue_map.png \
  30 40 50 60 70 80 90 100 \
  --in_labelmap tissue_map.nii.gz

python scil_screenshot_volume.py \
  t1.nii.gz \
  t1_axial_tissue_viridis_cmap.png \
  30 40 50 60 70 80 90 100 \
  --in_labelmap tissue_map.nii.gz \
  --labelmap_cmap_name viridis
"""

import argparse
import logging

from itertools import zip_longest
import numpy as np
from os.path import splitext

from scilpy.io.image import assert_same_resolution
from scilpy.io.utils import (
    get_default_screenshotting_data,
    add_nifti_screenshot_default_args,
    add_nifti_screenshot_overlays_args,
    assert_overlay_colors,
    add_verbose_arg,
    add_overwrite_arg,
    assert_inputs_exist
)
from scilpy.image.utils import check_slice_indices
from scilpy.viz.scene_utils import (
    compose_mosaic, screenshot_slice, screenshot_contour
)
from scilpy.viz.utils import RAS_AXES


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )

    add_nifti_screenshot_default_args(p, False, False)
    add_nifti_screenshot_overlays_args(p, transparency_is_overlay=False)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _parse_args(parser):

    args = parser.parse_args()

    inputs = []

    inputs.append(args.in_volume)

    if args.in_masks:
        inputs.extend(args.in_masks)
    if args.in_labelmap:
        inputs.append(args.in_labelmap)

    assert_inputs_exist(parser, inputs)

    assert_same_resolution(inputs)

    assert_overlay_colors(args.masks_colors, args.in_masks, parser)

    return args


def main():
    parser = _build_arg_parser()
    args = _parse_args(parser)
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    vol_img, t_mask_img, labelmap_img, mask_imgs, mask_colors = \
        get_default_screenshotting_data(args)

    # Check if the screenshots can be taken
    if args.slice_ids:
        check_slice_indices(vol_img, args.axis_name, args.slice_ids)
        slice_ids = args.slice_ids
    else:
        slice_ids = np.arange(vol_img.shape[RAS_AXES[args.axis_name]])

    # Generate the image slices
    vol_scene_container = screenshot_slice(
        vol_img,
        args.axis_name,
        slice_ids,
        args.win_dims
    )

    transparency_scene_container = []
    if t_mask_img is not None:
        transparency_scene_container = screenshot_slice(
            t_mask_img, args.axis_name, slice_ids, args.win_dims)

    labelmap_scene_container = []
    if labelmap_img:
        labelmap_scene_container = screenshot_slice(
            labelmap_img, args.axis_name, slice_ids, args.win_dims)

    mask_screenshotter = screenshot_slice
    if args.masks_as_contours:
        mask_screenshotter = screenshot_contour

    mask_overlay_scene_container, mask_overlay_colors = [], []
    if mask_imgs is not None:
        mask_overlay_colors = mask_colors
        for mask_img in mask_imgs:
             mask_overlay_scene_container.append(
                mask_screenshotter(
                    mask_img, args.axis_name, slice_ids, args.win_dims))

        mask_overlay_scene_container = np.swapaxes(
            mask_overlay_scene_container, 0, 1)

    name, ext = splitext(args.out_fname)
    names = ["{}_slice_{}{}".format(name, s, ext) for s in slice_ids]

    # Compose and save each slice
    for (volume, trans, label, contour, name, slice_id) in list(
        zip_longest(vol_scene_container,
                    transparency_scene_container,
                    labelmap_scene_container,
                    mask_overlay_scene_container,
                    names,
                    slice_ids,
                    fillvalue=tuple())):

        img = compose_mosaic(
            [volume], args.win_dims, 1, 1, [slice_id],
            vol_cmap_name=args.volume_cmap_name,
            transparency_scene_container=[trans],
            mask_overlay_scene_container=[contour],
            mask_overlay_color=mask_overlay_colors,
            mask_overlay_alpha=args.masks_alpha,
            labelmap_scene_container=[label],
            labelmap_cmap_name=args.labelmap_cmap_name,
            labelmap_overlay_alpha=args.labelmap_alpha,
            display_slice_number=args.display_slice_number,
            display_lr=args.display_lr)

        # Save the snapshot
        img.save(name)


if __name__ == "__main__":
    main()
