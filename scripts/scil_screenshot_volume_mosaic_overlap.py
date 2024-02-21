#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compose a mosaic of screenshots of the given image volume slices along the
requested axis. The provided transparency mask (e.g. a brain mask volume) is
used to set the screenshot values outside the mask non-zero values to full
transparency. Additionally, if a labelmap image is provided (e.g. a tissue
segmentation map), it is overlaid on the volume slices. Also, a series of 
masks can be provided and will be used to generate contours overlaid on each 
volume slice.

A labelmap image can be provided as the image volume, without requiring it as
the optional argument if only the former needs to be plot.

The screenshots are overlapped according to the given factors.

The mosaic supports either horizontal, vertical or matrix arrangements.

Example:
python scil_screenshot_volume_mosaic_overlap.py \
  t1.nii.gz \
  brain_mask.nii.gz \
  mosaic_overlap_t1_axial.png \
  30 40 50 60 70 80 90 100 \
  1 8

python scil_screenshot_volume_mosaic_overlap.py \
  t1.nii.gz \
  brain_mask.nii.gz \
  mosaic_overlap_t1_axial_plasma_cmap.png \
  30 40 50 60 70 80 90 100 \
  2 4 \
  --overlap_factor 0.6 0.5 \
  --vol_cmap_name plasma

python scil_screenshot_volume_mosaic_overlap.py \
  tissue_map.nii.gz \
  brain_mask.nii.gz \
  mosaic_overlap_tissue_axial_plasma_cmap.png \
  30 40 50 60 70 80 90 100 \
  2 4 \
  --vol_cmap_name plasma

python scil_screenshot_volume_mosaic_overlap.py \
  t1.nii.gz \
  brain_mask.nii.gz \
  mosaic_overlap_t1_sagittal_tissue_viridis_cmap.png \
  30 40 50 60 70 80 90 100 \
  2 4 \
  --axis_name sagittal \
  --in_labelmap tissue_map.nii.gz \
  --labelmap_cmap_name viridis

python scil_screenshot_volume_mosaic_overlap.py \
  t1.nii.gz \
  brain_mask.nii.gz \
  mosaic_overlap_t1_sagittal_tissue_contours.png \
  30 40 50 60 70 80 90 100 \
  2 4 \
  --axis_name sagittal \
  --in_contour_masks wm_mask.nii.gz gm_mask.nii.gz csf_mask.nii.gz
"""

import argparse
import logging

import numpy as np

from scilpy.io.image import assert_same_resolution
from scilpy.io.utils import (
    assert_overlay_colors,
    get_default_screenshotting_data,
    add_nifti_screenshot_default_args,
    add_nifti_screenshot_overlays_args,
    add_overwrite_arg,
    add_verbose_arg,
    assert_inputs_exist,
    assert_outputs_exist,
    ranged_type
)
from scilpy.image.utils import check_slice_indices
from scilpy.viz.scene_utils import (
    check_mosaic_layout, compose_mosaic, screenshot_contour, screenshot_slice
)


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )

    add_nifti_screenshot_default_args(p)
    add_nifti_screenshot_overlays_args(p, transparency_is_overlay=False)

    p.add_argument(
        "mosaic_rows_cols",
        nargs=2,
        # metavar=("rows", "cols"),  # CPython issue 58282
        type=int,
        help="The mosaic row and column count."
    )

    p.add_argument(
        "--overlap_factor",
        nargs=2,
        metavar=("OVERLAP_HORIZ", "OVERLAP_VERT"),
        default=(0.6, 0.0),
        type=ranged_type(float, 0.0, 1.0),
        help="The overlap factor with respect to the dimension. [%(default)s]"
    )

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _parse_args(parser):

    args = parser.parse_args()

    inputs = []
    output = []

    inputs.append(args.in_volume)
    inputs.append(args.in_transparency_mask)

    if args.in_labelmap:
        inputs.append(args.in_labelmap)
    if args.in_masks:
        inputs.extend(args.in_masks)

    output.append(args.out_fname)

    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, output)

    assert_same_resolution(inputs)

    assert_overlay_colors(args.masks_colors, args.in_masks, parser)

    return args


def main():
    parser = _build_arg_parser()
    args = _parse_args(parser)
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    vol_img, t_mask_img, labelmap_img, mask_imgs, mask_colors = \
        get_default_screenshotting_data(args)

    rows = args.mosaic_rows_cols[0]
    cols = args.mosaic_rows_cols[1]

    # Check if the mosaic can be built
    check_slice_indices(vol_img, args.axis_name, args.slice_ids)
    check_mosaic_layout(len(args.slice_ids), rows, cols)

    # Generate the images
    vol_scene_container = screenshot_slice(
        vol_img, args.axis_name, args.slice_ids, args.win_dims)

    transparency_scene_container = screenshot_slice(
        t_mask_img, args.axis_name, args.slice_ids, args.win_dims)

    labelmap_scene_container = []
    if labelmap_img:
        labelmap_scene_container = screenshot_slice(
            labelmap_img, args.axis_name, args.slice_ids, args.win_dims)

    mask_screenshotter = screenshot_slice
    if args.masks_as_contours:
        mask_screenshotter = screenshot_contour

    masks_scene_container = []
    if mask_imgs:
        for img in mask_imgs:
            masks_scene_container.append(
                mask_screenshotter(
                    img, args.axis_name, args.slice_ids, args.win_dims))

        masks_scene_container = np.swapaxes(masks_scene_container, 0, 1)

    # Compose the mosaic
    img = compose_mosaic(
        vol_scene_container, args.win_dims, rows, cols, args.slice_ids,
        overlap_factor=args.overlap_factor,
        vol_cmap_name=args.volume_cmap_name,
        transparency_scene_container=transparency_scene_container,
        labelmap_scene_container=labelmap_scene_container,
        labelmap_cmap_name=args.labelmap_cmap_name,
        labelmap_overlay_alpha=args.labelmap_alpha,
        mask_overlay_scene_container=masks_scene_container,
        mask_overlay_color=mask_colors,
        mask_overlay_alpha=args.masks_alpha)

    # Save the mosaic
    img.save(args.out_fname)


if __name__ == "__main__":
    main()
