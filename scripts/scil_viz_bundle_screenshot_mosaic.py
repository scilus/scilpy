#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize bundles from a list. The script will output a mosaic (image) with
screenshots, 6 views per bundle in the list.
"""

import argparse
import logging
import os
import random

from fury import actor, window
import nibabel as nib
import numpy as np

from PIL import Image
from PIL import ImageDraw

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             assert_headers_compatible)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.viz.backends.pil import fetch_truetype_font


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_volume',
                   help='Volume used as background (e.g. T1, FA, b0).')
    p.add_argument('in_bundles', nargs='+',
                   help='List of tractography files supported by nibabel '
                        'or binary mask files.')
    p.add_argument('out_image',
                   help='Name of the output image mosaic '
                        '(e.g. mosaic.jpg, mosaic.png).')

    p.add_argument('--uniform_coloring', nargs=3,
                   metavar=('R', 'G', 'B'), type=float,
                   help='Assign an uniform color to streamlines (or ROIs).')
    p.add_argument('--random_coloring', metavar='SEED', type=int,
                   help='Assign a random color to streamlines (or ROIs).')
    p.add_argument('--zoom', type=float, default=1.0,
                   help='Rendering zoom. '
                        'A value greater than 1 is a zoom-in,\n'
                        'a value less than 1 is a zoom-out [%(default)s].')

    p.add_argument('--ttf', default=None,
                   help='Path of the true type font to use for legends.')
    p.add_argument('--ttf_size', type=int, default=35,
                   help='Font size (int) to use for the legends '
                        '[%(default)s].')
    p.add_argument('--opacity_background', type=float, default=0.4,
                   help='Opacity of background image, between 0 and 1.0 '
                        '[%(default)s].')
    p.add_argument('--resolution_of_thumbnails', type=int, default=300,
                   help='Resolution of thumbnails used in mosaic '
                        '[%(default)s].')

    p.add_argument('--light_screenshot', action='store_true',
                   help='Keep only 3 views instead of 6 '
                        '[%(default)s].')
    p.add_argument('--no_information', action='store_true',
                   help='Don\'t display axis and bundle information '
                        '[%(default)s].')
    p.add_argument('--no_bundle_name', action='store_true',
                   help='Don\'t display bundle name '
                        '[%(default)s].')
    p.add_argument('--no_streamline_number', action='store_true',
                   help='Don\'t display bundle streamlines number '
                        '[%(default)s].')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def draw_column_with_names(draw, output_names, cell_width, cell_height,
                           font, row_count, no_bundle_name, no_elements_count):
    """
    Draw the first column with row's description
    (views and bundle information to display).
    """
    cell_half_width = cell_width / 2.
    cell_half_height = cell_height / 2.

    # Orientation's names
    for num, name in enumerate(output_names):
        j = cell_height * num
        draw.multiline_text((cell_half_width, j + cell_half_height),
                            name.replace('_', '\n'), font=font,
                            anchor='mm', align='center')

    # First column, last row: description of the information to show
    if not (no_bundle_name and no_elements_count):
        text = []
        if not no_bundle_name:
            text.append('Bundle')
        if not no_elements_count:
            text.append('Elements')

        j = cell_height * row_count
        padding = np.clip(cell_width // 10, 1, font.size)
        spacing = np.clip(cell_height // 10, 1, font.size)
        draw.multiline_text((cell_width - padding, j + cell_half_height),
                            '\n'.join(text), font=font, anchor='rm',
                            align='right', spacing=spacing)


def draw_bundle_information(draw, bundle_name, nbr_of_elem, col_center,
                            cell_height, font, view_number):
    """ Draw text with bundle information. """
    row_center = cell_height / 2.
    spacing = np.clip(cell_height // 10, 1, font.size)
    draw.multiline_text((col_center, cell_height * view_number + row_center),
                        "\n".join(filter(None, [bundle_name, nbr_of_elem])),
                        font=font, anchor='mm', align='center',
                        spacing=spacing)


def set_img_in_cell(mosaic, ren, view_number, width, height, i):
    """ Set a snapshot of the bundle in a cell of mosaic """

    out = window.snapshot(ren, size=(width, height))
    j = height * view_number
    # fury-gl flips image
    image = Image.fromarray(out[::-1])
    image.thumbnail((width, height))
    mosaic.paste(image, (i, j))


def random_rgb():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return np.array([r, g, b]) / 255.0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bundles + [args.in_volume],
                        args.reference)
    assert_outputs_exist(parser, args, args.out_image)
    assert_headers_compatible(parser, args.in_bundles + [args.in_volume],
                              reference=args.reference)

    output_names = ['axial_superior', 'axial_inferior',
                    'coronal_posterior', 'coronal_anterior',
                    'sagittal_left', 'sagittal_right']
    if args.light_screenshot:
        output_names = ['axial_superior',
                        'coronal_posterior',
                        'sagittal_left']

    output_dir = os.path.dirname(args.out_image)
    if output_dir:
        assert_output_dirs_exist_and_empty(parser, args, output_dir,
                                           create_dir=True)

    # ----------------------------------------------------------------------- #
    # Mosaic, column 0: orientation names and data description
    # ----------------------------------------------------------------------- #
    width = args.resolution_of_thumbnails
    height = args.resolution_of_thumbnails
    cell_half_width = width / 2.

    rows = 6
    if args.light_screenshot:
        rows = 3
    cols = len(args.in_bundles)

    # Creates a new empty image, RGB mode
    if args.no_information:
        mosaic = Image.new('RGB', (cols * width, rows * height))
    elif args.no_bundle_name and args.no_streamline_number:
        mosaic = Image.new('RGB', ((cols + 1) * width, rows * height))
    else:
        mosaic = Image.new('RGB', ((cols + 1) * width, (rows + 1) * height))

    # Prepare draw and font objects to render text
    draw = ImageDraw.Draw(mosaic)
    font = fetch_truetype_font(args.ttf or "freesans", args.ttf_size)

    # Data of the volume used as background
    ref_img = nib.load(args.in_volume)
    data = ref_img.get_fdata(dtype=np.float32)
    affine = ref_img.affine
    mean, std = data[data > 0].mean(), data[data > 0].std()
    value_range = (mean - 0.5 * std, mean + 1.5 * std)

    # First column with rows description
    if not args.no_information:
        draw_column_with_names(draw, output_names, width, height, font, rows,
                               args.no_bundle_name, args.no_streamline_number)

    # ----------------------------------------------------------------------- #
    # Columns with bundles
    # ----------------------------------------------------------------------- #
    random.seed(args.random_coloring)
    for idx_bundle, bundle_file in enumerate(args.in_bundles):

        bundle_file_name = os.path.basename(bundle_file)
        bundle_name, bundle_ext = split_name_with_nii(bundle_file_name)

        if args.no_information:
            i = idx_bundle * width
        else:
            i = (idx_bundle + 1) * width

        if not os.path.isfile(bundle_file) and not args.no_information:
            logging.warning(
                '\nInput file {} doesn\'t exist.'.format(bundle_file))

            number_streamlines = 0
        else:
            if args.uniform_coloring:
                colors = args.uniform_coloring
            elif args.random_coloring is not None:
                colors = random_rgb()
            # Select the streamlines to plot
            if bundle_ext in ['.tck', '.trk']:
                if (args.random_coloring is None
                        and args.uniform_coloring is None):
                    colors = None
                bundle_tractogram_file = nib.streamlines.load(bundle_file)
                streamlines = bundle_tractogram_file.streamlines
                if len(streamlines):
                    bundle_actor = actor.line(streamlines, colors)
                number_streamlines = str(len(streamlines))
            # Select the volume to plot
            elif bundle_ext in ['.nii.gz', '.nii']:
                if not args.random_coloring and not args.uniform_coloring:
                    colors = [1.0, 1.0, 1.0]
                bundle_img_file = nib.load(bundle_file)
                roi = get_data_as_mask(bundle_img_file)
                bundle_actor = actor.contour_from_roi(roi,
                                                      bundle_img_file.affine,
                                                      colors)
                number_streamlines = str(np.count_nonzero(roi))

            # Render
            ren = window.Scene()
            zoom = args.zoom
            opacity = args.opacity_background

            # Structural data
            slice_actor = actor.slicer(data, affine, value_range)
            slice_actor.opacity(opacity)
            ren.add(slice_actor)

            # Streamlines
            if len(streamlines):
                ren.add(bundle_actor)
            ren.reset_camera()
            ren.zoom(zoom)
            view_number = 0
            set_img_in_cell(mosaic, ren, view_number, width, height, i)

            if not args.light_screenshot:
                ren.pitch(180)
                ren.reset_camera()
                ren.zoom(zoom)
                view_number += 1
                set_img_in_cell(mosaic, ren, view_number, width, height, i)

            ren.rm(slice_actor)
            slice_actor2 = slice_actor.copy()
            slice_actor2.display(None, slice_actor2.shape[1]//2, None)
            slice_actor2.opacity(opacity)
            ren.add(slice_actor2)

            ren.pitch(90)
            ren.set_camera(view_up=(0, 0, 1))
            ren.reset_camera()
            ren.zoom(zoom)
            view_number += 1
            set_img_in_cell(mosaic, ren, view_number, width, height, i)

            if not args.light_screenshot:
                ren.pitch(180)
                ren.set_camera(view_up=(0, 0, 1))
                ren.reset_camera()
                ren.zoom(zoom)
                view_number += 1
                set_img_in_cell(mosaic, ren, view_number, width, height, i)

            ren.rm(slice_actor2)
            slice_actor3 = slice_actor.copy()
            slice_actor3.display(slice_actor3.shape[0]//2, None, None)
            slice_actor3.opacity(opacity)
            ren.add(slice_actor3)

            ren.yaw(90)
            ren.reset_camera()
            ren.zoom(zoom)
            view_number += 1
            set_img_in_cell(mosaic, ren, view_number, width, height, i)

            if not args.light_screenshot:
                ren.yaw(180)
                ren.reset_camera()
                ren.zoom(zoom)
                view_number += 1
                set_img_in_cell(mosaic, ren, view_number, width, height, i)

        if not args.no_information:
            if args.no_bundle_name:
                bundle_name = None
            if args.no_streamline_number:
                number_streamlines = None
            if not (args.no_bundle_name and args.no_streamline_number):
                draw_bundle_information(draw, bundle_name, number_streamlines,
                                        i + cell_half_width, height,
                                        font, rows)

    if not args.no_information:
        j = rows * height
        draw.line([(width, 0), (width, j)], fill='white')
        if not (args.no_bundle_name and args.no_streamline_number):
            draw.line([(0, j), ((cols + 1) * width, j)], fill='white')

    # Save image to file
    mosaic.save(args.out_image)


if __name__ == '__main__':
    main()
