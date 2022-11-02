#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compose a mosaic of screenshots of the given image volume slices along the
requested axis. The provided transparency mask (e.g. a brain mask volume) is
used to set the screenshot values outside the mask non-zero values to full
transparency. Additionally, if a labelmap image is provided (e.g. a tissue
segmentation map), it is overlaid on the volume slices.

A labelmap image can be provided as the image volume, without requiring it as
the optional argument if only the former needs to be plot.

The screenshots are overlapped according to the given factors.

The mosaic supports either horizontal, vertical or matrix arrangements.

Example:
python scil_screenshot_volume_mosaic_overlap.py \
  t1.nii.gz \
  brain_mask.nii.gz \
  mosaic_overlap_t1.png \
  30 40 50 60 70 80 90 100 \
  1 8

python scil_screenshot_volume_mosaic_overlap.py \
  t1.nii.gz \
  brain_mask.nii.gz \
  mosaic_overlap_t1_matrix_cmap.png \
  30 40 50 60 70 80 90 100 \
  2 4 \
  --overlap_factor 0.6 0.5 \
  --vol_cmap_name plasma

python scil_screenshot_volume_mosaic_overlap.py \
  tissue_map.nii.gz \
  brain_mask.nii.gz \
  mosaic_overlap_tissue_map.png \
  30 40 50 60 70 80 90 100 \
  2 4 \
  --vol_cmap_name plasma

python scil_screenshot_volume_mosaic_overlap.py \
  t1.nii.gz \
  brain_mask.nii.gz \
  mosaic_overlap_t1_tisue_map.png \
  30 40 50 60 70 80 90 100 \
  2 4 \
  --in_labelmap tissue_map.nii.gz \
  --axis_name sagittal \
  --labelmap_cmap_name viridis
"""

import argparse

import nibabel as nib
from dipy.io.utils import is_header_compatible

from scilpy.io.image import assert_same_resolution
from scilpy.io.utils import (
    axis_name_choices,
    add_overwrite_arg,
    assert_inputs_exist,
    assert_outputs_exist,
    ranged_type,
)
from scilpy.image.utils import check_slice_indices
from scilpy.viz.scene_utils import (
    check_mosaic_layout, compose_mosaic, screenshot_slice,
)


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Positional arguments
    p.add_argument("in_vol", help="Input volume image file.")
    p.add_argument("in_transparency_mask", help="Input mask image file.")
    p.add_argument(
        "out_fname",
        help="Name of the output image mosaic (e.g. mosaic.jpg, mosaic.png)."
    )
    p.add_argument(
        "slice_ids", nargs="+", type=int, help="Slice indices for the mosaic."
    )
    p.add_argument(
        "mosaic_rows_cols",
        nargs=2,
        # metavar=("rows", "cols"),  # CPython issue 58282
        type=int,
        help="The mosaic row and column count."
    )

    # Optional arguments
    p.add_argument("--in_labelmap",  help="Labelmap image.")
    p.add_argument(
        "--axis_name",
        default="axial",
        type=str,
        choices=axis_name_choices,
        help="Name of the axis to visualize. [%(default)s]"
    )
    p.add_argument(
        "--overlap_factor",
        nargs=2,
        metavar=("OVERLAP_HORIZ", "OVERLAP_VERT"),
        default=(0.6, 0.0),
        type=ranged_type(float, 0.0, 1.0),
        help="The overlap factor with respect to the dimension. [%(default)s]"
    )
    p.add_argument(
        "--win_dims",
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(768, 768),
        type=int,
        help="The dimensions for the vtk window. [%(default)s]"
    )
    p.add_argument(
        "--vol_cmap_name",
        default=None,
        help="Colormap name for the volume image data. [%(default)s]"
    )
    p.add_argument(
        "--labelmap_cmap_name",
        default="viridis",
        help="Colormap name for the labelmap image data. [%(default)s]"
    )

    add_overwrite_arg(p)

    return p


def _parse_args(parser):

    args = parser.parse_args()

    inputs = []
    output = []

    inputs.append(args.in_vol)
    inputs.append(args.in_transparency_mask)

    if args.in_labelmap:
        inputs.append(args.in_labelmap)

    output.append(args.out_fname)

    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, output)

    assert_same_resolution(inputs)

    return args


def _get_data_from_inputs(args):

    vol_img = nib.load(args.in_vol)
    mask_img = nib.load(args.in_transparency_mask)

    # Check header compatibility
    if not is_header_compatible(vol_img, mask_img):
        raise ValueError(
            f"{args.in_vol} and {args.in_mask} do not have a compatible "
            f"header."
        )

    labelmap_img = None
    if args.in_labelmap:
        labelmap_img = nib.load(args.in_labelmap)

        # Check header compatibility
        if not is_header_compatible(vol_img, labelmap_img):
            raise ValueError(
                f"{args.in_vol} and {args.in_labelmap} do not have a "
                f"compatible header."
            )

    return vol_img, mask_img, labelmap_img


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    vol_img, mask_img, labelmap_img = _get_data_from_inputs(args)

    rows = args.mosaic_rows_cols[0]
    cols = args.mosaic_rows_cols[1]

    # Check if the mosaic can be built
    check_slice_indices(vol_img, args.axis_name, args.slice_ids)
    check_mosaic_layout(len(args.slice_ids), rows, cols)

    # Generate the images
    vol_scene_container = screenshot_slice(
        vol_img,
        args.axis_name,
        args.slice_ids,
        args.win_dims,
    )
    mask_scene_container = screenshot_slice(
        mask_img,
        args.axis_name,
        args.slice_ids,
        args.win_dims,
    )

    labelmap_scene_container = []
    if labelmap_img:
        labelmap_scene_container = screenshot_slice(
            labelmap_img,
            args.axis_name,
            args.slice_ids,
            args.win_dims,
        )

    # Compose the mosaic
    img = compose_mosaic(
        vol_scene_container,
        mask_scene_container,
        args.win_dims,
        rows,
        cols,
        args.overlap_factor,
        labelmap_scene_container=labelmap_scene_container,
        vol_cmap_name=args.vol_cmap_name,
        labelmap_cmap_name=args.labelmap_cmap_name,
        )

    # Save the mosaic
    img.save(args.out_fname)


if __name__ == "__main__":
    main()
