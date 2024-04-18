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
>>> scil_viz_volume_screenshot_mosaic.py 1 8 t1.nii.gz brain_mask.nii.gz
    mosaic_overlap_t1_axial.png 30 40 50 60 70 80 90 100

>>> scil_viz_volume_screenshot_mosaic.py 2 4 t1.nii.gz brain_mask.nii.gz
    mosaic_overlap_t1_axial_plasma_cmap.png 30 40 50 60 70 80 90 100
    --overlap_factor 0.6 0.5 --volume_cmap_name plasma

>>> scil_viz_volume_screenshot_mosaic.py 2 4 tissues.nii.gz brain_mask.nii.gz
    mosaic_overlap_tissues_axial_plasma_cmap.png 30 40 50 60 70 80 90 100
    --volume_cmap_name plasma

>>> scil_viz_volume_screenshot_mosaic.py 2 4 t1.nii.gz brain_mask.nii.gz
    mosaic_overlap_t1_sagittal_tissue_viridis_cmap.png
    30 40 50 60 70 80 90 100 --axis sagittal
    --labelmap tissue_map.nii.gz --labelmap_cmap_name viridis

>>> scil_viz_volume_screenshot_mosaic.py 2 4 t1.nii.gz brain_mask.nii.gz
    mosaic_overlap_t1_sagittal_tissue_contours.png
    30 40 50 60 70 80 90 100 --axis sagittal
    --overlays wm_mask.nii.gz gm_mask.nii.gz csf_mask.nii.gz
"""

import argparse
from functools import partial
import itertools
import logging

from scilpy.io.utils import (add_default_screenshot_args,
                             add_labelmap_screenshot_args,
                             add_overlays_screenshot_args,
                             add_overwrite_arg,
                             add_verbose_arg, assert_headers_compatible,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_overlay_colors,
                             get_default_screenshotting_data,
                             ranged_type)
from scilpy.image.utils import check_slice_indices
from scilpy.viz.screenshot import (compose_mosaic,
                                   screenshot_contour,
                                   screenshot_volume)
from scilpy.viz.utils import check_mosaic_layout


def _build_arg_parser():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    xg = p.add_argument_group(title="Input overlays")
    vg = p.add_argument_group(title="Volume rendering")
    og = p.add_argument_group(title="Overlay rendering")

    p.add_argument("rows", type=int, help="The mosaic row count.")
    p.add_argument("cols", type=int, help="The mosaic column count.")

    add_default_screenshot_args(p, disable_annotations=True,
                                cmap_parsing_group=vg,
                                opacity_parsing_group=vg)

    add_labelmap_screenshot_args(xg, "viridis", 0.5, vg, vg)
    add_overlays_screenshot_args(xg, 0.5, og)

    p.add_argument("--overlap", nargs=2, metavar=("rWIDTH", "rHEIGHT"),
                   default=(0.6, 0.0), type=ranged_type(float, 0.0, 1.0),
                   help="The overlap factor as a ratio of each image "
                        "dimension. [%(default)s]")

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _parse_args(parser):
    args = parser.parse_args()

    required = [args.volume, args.transparency]
    output = [args.out_fname]
    optional = []

    if args.overlays:
        optional.extend(args.overlays)
    if args.labelmap:
        optional.append(args.labelmap)

    assert_inputs_exist(parser, required, optional)
    assert_outputs_exist(parser, args, output)
    assert_headers_compatible(parser, required, optional)
    assert_overlay_colors(args.overlays_colors, args.overlays, parser)

    return args


def main():
    def empty_generator():
        yield from ()

    parser = _build_arg_parser()
    args = _parse_args(parser)
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    vol_img, trans_img, labelmap_img, ovl_imgs, ovl_colors, _ = \
        get_default_screenshotting_data(args, peaks=False)

    rows, cols = args.rows, args.cols

    # Check if the mosaic can be built
    check_slice_indices(vol_img, args.axis, args.slices)
    check_mosaic_layout(len(args.slices), rows, cols)

    # Generate the images
    volume_screenshots_generator = screenshot_volume(
        vol_img, args.axis, args.slices, args.size)

    transparency_screenshots_generator = screenshot_volume(
        trans_img, args.axis, args.slices, args.size)

    labelmap_screenshots_generator = empty_generator()
    if labelmap_img:
        labelmap_screenshots_generator = screenshot_volume(
            labelmap_img, args.axis, args.slices, args.size)

    # Create the overlay screenshotter
    overlay_screenshotter = screenshot_volume
    if args.overlays_as_contours:
        overlay_screenshotter = partial(screenshot_contour,
                                        bg_opacity=0.3)

    # Generate the overlay stack, if requested, zipping over all overlays
    overlay_screenshots_generator = empty_generator()
    if ovl_imgs is not None:
        overlay_screenshots_generator = zip(*itertools.starmap(
            overlay_screenshotter, ([ovl, args.axis, args.slices,
                                     args.size] for ovl in ovl_imgs)))

    # Compose the mosaic
    img = compose_mosaic(
        volume_screenshots_generator, args.size, rows, cols, args.slices,
        overlap_factor=args.overlap,
        transparency_scene_container=transparency_screenshots_generator,
        image_alpha=args.volume_opacity,
        labelmap_scene_container=labelmap_screenshots_generator,
        labelmap_overlay_alpha=args.labelmap_opacity,
        overlays_scene_container=overlay_screenshots_generator,
        overlays_colors=ovl_colors,
        overlays_alpha=args.overlays_opacity)

    # Save the mosaic
    img.save(args.out_fname)


if __name__ == "__main__":
    main()
