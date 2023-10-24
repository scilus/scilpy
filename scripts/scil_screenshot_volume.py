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
import itertools
import numpy as np
from os.path import splitext

from scilpy.io.image import assert_same_resolution
from scilpy.io.utils import (add_nifti_screenshot_default_args,
                             add_nifti_screenshot_overlays_args,
                             add_nifti_screenshot_peaks_arg,
                             add_verbose_arg,
                             add_overwrite_arg,
                             get_default_screenshotting_data,
                             assert_inputs_exist,
                             assert_overlay_colors)
from scilpy.image.utils import check_slice_indices
from scilpy.utils.util import get_axis_index
from scilpy.viz.screenshot import (compose_image,
                                   screenshot_contour,
                                   screenshot_peaks,
                                   screenshot_volume)


def _build_arg_parser():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    add_nifti_screenshot_default_args(p, False, False)
    add_nifti_screenshot_overlays_args(p, transparency_is_overlay=False)
    add_nifti_screenshot_peaks_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _parse_args(parser):

    args = parser.parse_args()
    inputs = [args.in_volume]

    if args.in_masks:
        inputs.extend(args.in_masks)
    if args.in_labelmap:
        inputs.append(args.in_labelmap)
    if args.in_peaks:
        inputs.append(args.in_peaks)

    assert_inputs_exist(parser, inputs)
    assert_same_resolution(inputs)
    assert_overlay_colors(args.masks_colors, args.in_masks, parser)

    # TODO : check outputs (we need to know the slicing), could be glob

    return args


def main():
    def empty_generator():
        yield from ()

    parser = _build_arg_parser()
    args = _parse_args(parser)
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    vol_img, t_mask_img, labelmap_img, mask_imgs, mask_colors, peaks_imgs = \
        get_default_screenshotting_data(args)

    # Check if the screenshots can be taken
    if args.slice_ids:
        check_slice_indices(vol_img, args.axis_name, args.slice_ids)
        slice_ids = args.slice_ids
    else:
        ax_idx = get_axis_index(args.axis_name)
        slice_ids = np.arange(vol_img.shape[ax_idx])

    # Generate the image slices
    volume_screenhots_generator = screenshot_volume(vol_img, args.axis_name,
                                                    slice_ids, args.win_dims)

    # Generate transparency, if requested
    transparency_screenshots_generator = empty_generator()
    if t_mask_img is not None:
        transparency_screenshots_generator = screenshot_volume(
            t_mask_img, args.axis_name, slice_ids, args.win_dims)

    # Generate labelmap, if requested
    labelmap_screenshots_generator = empty_generator()
    if labelmap_img:
        labelmap_screenshots_generator = screenshot_volume(
            labelmap_img, args.axis_name, slice_ids, args.win_dims)

    # Create the overlay screenshotter
    overlay_screenshotter = screenshot_volume
    if args.masks_as_contours:
        def _dual_screenshot(*args, **kwargs):
            return screenshot_contour(*args, **kwargs, bg_opacity=0.3)

        overlay_screenshotter = _dual_screenshot

    # Generate the overlay stack, if requested, zipping over all overlays
    overlay_screenshots_generator, mask_overlay_colors = empty_generator(), []
    if mask_imgs is not None:
        mask_overlay_colors = mask_colors
        overlay_screenshots_generator = zip(*itertools.starmap(
            overlay_screenshotter, ([mask, args.axis_name, slice_ids,
                                     args.win_dims] for mask in mask_imgs)))

    if peaks_imgs is not None:
        peaks_screenshots_generator = zip(*itertools.starmap(
            screenshot_peaks, ([peaks, args.axis_name, slice_ids,
                                args.win_dims] for peaks in peaks_imgs)))

    name, ext = splitext(args.out_fname)
    names = ["{}_slice_{}{}".format(name, s, ext) for s in slice_ids]
    sides_labels = ["A", "P"] if args.axis_name == "sagittal" else ["L", "R"]

    # Compose and save each slice
    for volume, trans, label, contour, peaks, name, slice_id in zip_longest(
            volume_screenhots_generator,
            transparency_screenshots_generator,
            labelmap_screenshots_generator,
            overlay_screenshots_generator,
            peaks_screenshots_generator,
            names,
            slice_ids,
            fillvalue=None):

        img = compose_image(volume, args.win_dims, slice_id,
                            vol_cmap_name=args.volume_cmap_name,
                            transparency_scene=trans,
                            mask_overlay_scene=contour,
                            mask_overlay_color=mask_overlay_colors,
                            mask_overlay_alpha=args.masks_alpha,
                            labelmap_scene=label,
                            labelmap_cmap_name=args.labelmap_cmap_name,
                            labelmap_overlay_alpha=args.labelmap_alpha,
                            peaks_overlay_scene=peaks,
                            peaks_overlay_alpha=args.peaks_alpha,
                            display_slice_number=args.display_slice_number,
                            display_lr=args.display_lr,
                            lr_labels=sides_labels)

        # Save the snapshot
        img.save(name)


if __name__ == "__main__":
    main()
