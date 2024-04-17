#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Take screenshot(s) of one or more slices in a given image volume along the
requested axis. If slice indices are not provided, all slices in the volume
are used. The name of the output images are suffixed with _slice_{id}, with
id being the slice number in the volume. If a labelmap image is provided (e.g.
a tissue segmentation map), it is overlaid on the volume slices. Same goes if
a mask is provided, with the difference that it can be rendered as a
transparency overlay as well as a contour.

A labelmap image can be provided as the image volume, without requiring it as
the optional argument if only the former needs to be plot.

Example:
>>> scil_viz_volume_screenshot.py t1.nii.gz t1_axial_annotated.png
    --display_slice_number --display_lr

>>> scil_viz_volume_screenshot.py t1.nii.gz t1_axial_masked.png
    --transparency brainmask.nii.gz

>>> scil_viz_volume_screenshot.py t1.nii.gz t1_axial.png
    --slices 30 40 50 60 70 80 90 100

>>> scil_viz_volume_screenshot.py t1.nii.gz t1_sagittal.png --axis sagittal

>>> scil_viz_volume_screenshot.py t1.nii.gz t1_axial_plasma_cmap.png
    --slices 30 40 50 60 70 80 90 100 --volume_cmap_name plasma

>>> scil_viz_volume_screenshot.py t1.nii.gz t1_mask_overlay.png
    --slices 30 40 50 60 70 80 90 100 --overlays brain_mask.nii.gz

>>> scil_viz_volume_screenshot.py t1.nii.gz t1_mask_contour.png
    --slices 30 40 50 60 70 80 90 100
    --overlays brain_mask.nii.gz --overlays_as_contours

>>> scil_viz_volume_screenshot.py t1.nii.gz t1_axial_tissue_map.png
    --slices 30 40 50 60 70 80 90 100 --labelmap tissue_map.nii.gz

>>> scil_viz_volume_screenshot.py t1.nii.gz t1_axial_tissue_viridis_cmap.png
    --slices 30 40 50 60 70 80 90 100
    --labelmap tissue_map.nii.gz --labelmap_cmap_name viridis

>>> scil_viz_volume_screenshot.py t1.nii.gz t1_axial_peaks.png
    --slices 30 40 50 60 70 80 90 100 --peaks peaks.nii.gz --volume_opacity 0.5
"""

import argparse
import logging

from functools import partial
from itertools import zip_longest
import itertools
import numpy as np
from os.path import splitext

from scilpy.io.utils import (add_default_screenshot_args,
                             add_labelmap_screenshot_args,
                             add_overlays_screenshot_args,
                             add_peaks_screenshot_args,
                             add_verbose_arg, assert_headers_compatible,
                             assert_inputs_exist,
                             assert_overlay_colors,
                             get_default_screenshotting_data)
from scilpy.image.utils import check_slice_indices
from scilpy.utils.spatial import get_axis_index
from scilpy.viz.screenshot import (compose_image,
                                   screenshot_contour,
                                   screenshot_peaks,
                                   screenshot_volume)


def _build_arg_parser():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    sg = p.add_argument_group(title="Slicing")
    xg = p.add_argument_group(title="Input overlays")
    vg = p.add_argument_group(title="Volume rendering")
    pg = p.add_argument_group(title="Peaks rendering")
    og = p.add_argument_group(title="Overlay rendering")
    ag = p.add_argument_group(title="Annotations")

    add_default_screenshot_args(p, False, False, False, sg, ag, vg, vg)
    add_labelmap_screenshot_args(xg, "viridis", 0.5, vg, vg)
    add_overlays_screenshot_args(xg, 0.5, og)
    add_peaks_screenshot_args(xg, rendering_parsing_group=pg)
    add_verbose_arg(p)

    return p


def _parse_args(parser):

    args = parser.parse_args()

    required = [args.volume]
    optional = []

    if args.overlays:
        optional.extend(args.overlays)
    if args.peaks:
        optional.extend(args.peaks)
    if args.labelmap:
        optional.append(args.labelmap)
    if args.transparency:
        optional.append(args.transparency)

    assert_inputs_exist(parser, required, optional)
    assert_headers_compatible(parser, required, optional)
    assert_overlay_colors(args.overlays_colors, args.overlays, parser)

    return args


def main():
    def empty_generator():
        yield from ()

    parser = _build_arg_parser()
    args = _parse_args(parser)
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    vol_img, trans_img, labelmap_img, ovl_imgs, ovl_colors, peaks_imgs = \
        get_default_screenshotting_data(args)

    # Check if the screenshots can be taken
    if args.slices:
        check_slice_indices(vol_img, args.axis, args.slices)
        slice_ids = args.slices
    else:
        ax_idx = get_axis_index(args.axis)
        slice_ids = np.arange(vol_img.shape[ax_idx])

    # Generate the image slices
    volume_screenhots_generator = screenshot_volume(vol_img, args.axis,
                                                    slice_ids, args.size,
                                                    args.volume_cmap_name)

    # Generate transparency, if requested
    transparency_screenshots_generator = empty_generator()
    if trans_img is not None:
        transparency_screenshots_generator = screenshot_volume(
            trans_img, args.axis, slice_ids, args.size)

    # Generate labelmap, if requested
    labelmap_screenshots_generator = empty_generator()
    if labelmap_img:
        labelmap_screenshots_generator = screenshot_volume(
            labelmap_img, args.axis, slice_ids, args.size,
            args.labelmap_cmap_name)

    # Create the overlay screenshotter
    overlay_screenshotter = screenshot_volume
    overlay_alpha = args.overlays_opacity
    if args.overlays_as_contours:
        overlay_screenshotter = partial(screenshot_contour,
                                        bg_opacity=args.overlays_opacity)
        overlay_alpha = 1.0

    # Generate the overlay stack, if requested, zipping over all overlays
    overlay_screenshots_generator, overlays_colors = empty_generator(), []
    if ovl_imgs is not None:
        overlays_colors = ovl_colors
        overlay_screenshots_generator = zip(*itertools.starmap(
            overlay_screenshotter, ([ovl, args.axis, slice_ids,
                                     args.size] for ovl in ovl_imgs)))

    peaks_screenshots_generator = empty_generator()
    if peaks_imgs is not None:
        peaks_screenshots_generator = zip(*itertools.starmap(
            screenshot_peaks, ([peaks, args.axis, slice_ids,
                                args.size] for peaks in peaks_imgs)))

    name, ext = splitext(args.out_fname)
    names = ["{}_slice_{}{}".format(name, s, ext) for s in slice_ids]
    sides_labels = ["P", "A"] if args.axis == "sagittal" else ["L", "R"]

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

        img = compose_image(volume, args.size, slice_id,
                            transparency_scene=trans,
                            image_alpha=args.volume_opacity,
                            overlays_scene=contour,
                            overlays_colors=overlays_colors,
                            overlays_alpha=overlay_alpha,
                            labelmap_scene=label,
                            labelmap_overlay_alpha=args.labelmap_opacity,
                            peaks_overlay_scene=peaks,
                            peaks_overlay_alpha=args.peaks_opacity,
                            display_slice_number=args.display_slice_number,
                            display_lr=args.display_lr,
                            lr_labels=sides_labels)

        # Save the snapshot
        img.save(name)


if __name__ == "__main__":
    main()
