#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize bundles from a list.
The script will output a mosaic (image) with screenshots,
6 views per bundle in the list.
"""

from __future__ import division, print_function
import argparse
import logging
import os
import shutil

import nibabel as nib
from fury import actor, window

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exist)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('anat_reference',
                        help='Image used as background (e.g. T1, FA, b0).')
    parser.add_argument('inputs', nargs='+',
                        help='List of streamline files supported by nibabel.')
    parser.add_argument('output_name',
                        help='Name of the output image mosaic '
                             '(e.g. mosaic.jpg, mosaic.png).')
    parser.add_argument('--zoom', type=float, default=1.0,
                        help='Rendering zoom. '
                             'A value greater than 1 is a zoom-in, '
                             'a value less than 1 is a zoom-out.')
    parser.add_argument('--ttf', default=None,
                        help='Path of the true type font to use for legends.')
    parser.add_argument('--ttf_size', type=int, default=35,
                        help='Font size (int) to use for the legends.')
    parser.add_argument('--opacity_background', type=float, default=0.4,
                        help='Opacity of background image [0, 1.0]')
    parser.add_argument('--resolution_of_thumbnails', type=int, default=300,
                        help='Resolution of thumbnails used in mosaic.')

    add_overwrite_arg(parser)
    return parser


def get_font(args):
    """Returns a ttf font object."""
    if args.ttf is not None:
        try:
            font = ImageFont.truetype(args.ttf, args.ttf_size)
        except Exception:
            logging.error('Font {} was not found. '
                          'Default font will be used.'.format(args.ttf))
            font = ImageFont.load_default()
    elif args.ttf_size is not None:
        # default font is not a truetype font, so size can't be changed.
        # to allow users to change the size without having to know where fonts
        # are in their computer, we could try to find a truetype font
        # ourselves. They are often present in /usr/share/fonts/
        font_path = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
        try:
            font = ImageFont.truetype(font_path, args.ttf_size)
        except Exception:
            logging.error('You did not specify a font. It is difficult'
                          'for us to adjust size. We tried on font {} '
                          'but it was not found.'
                          'Default font will be used, for which font '
                          'cannot be changed.'.format(font_path))
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    return font


def draw_column_with_names(draw, output_names, text_pos_x,
                           text_pos_y, height, font):
    """
    Draw the first column with row's description
    (views and bundle information to display).
    """
    # Orientation's names
    for num, name in enumerate(output_names):
        i = 0
        j = height * num + 50
        # Name splited in two lines
        draw.text((i + text_pos_x, j + text_pos_y),
                  name[:name.find('_')], font=font)
        draw.text((i + text_pos_x, j + text_pos_y + font.getsize(' ')[1]*1.5),
                  name[1+name.find('_'):], font=font)

    # First column, last row: description of the information to show
    view_number = 6
    i = 0
    j = height * view_number
    draw.text((i + text_pos_x, j + text_pos_y),
              ('Bundle'), font=font)
    draw.text((i + text_pos_x, j + text_pos_y + font.getsize(' ')[1]*1.5),
              ('Streamlines'), font=font)


def draw_bundle_information(draw, bundle_file_name, number_streamlines,
                            pos_x, pos_y, font):
    """Draw text with bundle information."""
    draw.text((pos_x, pos_y),
              (bundle_file_name), font=font)
    draw.text((pos_x, pos_y + font.getsize(' ')[1]*1.5),
              ('{}'.format(number_streamlines)), font=font)


def set_img_in_cell(mosaic, ren, view_number, path, width, height, i):
    """Set a snapshot of the bundle in a cell of mosaic"""
    window.snapshot(ren, path, size=(width, height))
    j = height * view_number
    image = Image.open(path)
    image.thumbnail((width, height))
    mosaic.paste(image, (i, j))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.anat_reference])
    assert_outputs_exist(parser, args, [args.output_name])

    output_names = ['axial_superior', 'axial_inferior',
                    'coronal_posterior', 'coronal_anterior',
                    'sagittal_left', 'sagittal_right']

    list_of_bundles = [f for f in args.inputs]
    # output_dir: where temporary files will be created
    output_dir = os.path.dirname(args.output_name)

    # ----------------------------------------------------------------------- #
    # Mosaic, column 0: orientation names and data description
    # ----------------------------------------------------------------------- #
    width = args.resolution_of_thumbnails
    height = args.resolution_of_thumbnails
    rows = 6
    cols = len(list_of_bundles)
    text_pos_x = 50
    text_pos_y = 50

    # Creates a new empty image, RGB mode
    mosaic = Image.new('RGB', ((cols + 1) * width, (rows + 1) * height))

    # Prepare draw and font objects to render text
    draw = ImageDraw.Draw(mosaic)
    font = get_font(args)

    # Data of the image used as background
    ref_img = nib.load(args.anat_reference)
    data = ref_img.get_data()
    affine = ref_img.affine
    mean, std = data[data > 0].mean(), data[data > 0].std()
    value_range = (mean - 0.5 * std, mean + 1.5 * std)

    # First column with rows description
    draw_column_with_names(draw, output_names, text_pos_x,
                           text_pos_y, height, font)

    # ----------------------------------------------------------------------- #
    # Columns with bundles
    # ----------------------------------------------------------------------- #
    for idx_bundle, bundle_file in enumerate(list_of_bundles):

        bundle_file_name = os.path.basename(bundle_file)
        bundle_name, _ = os.path.splitext(bundle_file_name)

        # !! It creates a temporary folder to create
        # the images to concatenate in the mosaic !!
        output_bundle_dir = os.path.join(output_dir, bundle_name)
        if not os.path.isdir(output_bundle_dir):
            os.makedirs(output_bundle_dir)

        output_paths = [
            os.path.join(output_bundle_dir,
                         '{}_' + os.path.basename(
                             output_bundle_dir)).format(name)
            for name in output_names]

        i = (idx_bundle + 1)*width

        if not os.path.isfile(bundle_file):
            print('\nInput file {} doesn\'t exist.'.format(bundle_file))

            number_streamlines = 0

            view_number = 6
            j = height * view_number

            draw_bundle_information(draw, bundle_file_name, number_streamlines,
                                    i + text_pos_x, j + text_pos_y, font)

        else:
            # Select the streamlines to plot
            bundle_tractogram_file = nib.streamlines.load(bundle_file)
            streamlines = bundle_tractogram_file.streamlines

            tubes = actor.line(streamlines)

            number_streamlines = len(streamlines)

            # Render
            ren = window.Renderer()
            zoom = args.zoom
            opacity = args.opacity_background

            # Structural data
            slice_actor = actor.slicer(data, affine, value_range)
            slice_actor.opacity(opacity)
            ren.add(slice_actor)

            # Streamlines
            ren.add(tubes)
            ren.reset_camera()
            ren.zoom(zoom)
            view_number = 0
            set_img_in_cell(mosaic, ren, view_number,
                            output_paths[view_number], width, height, i)

            ren.pitch(180)
            ren.reset_camera()
            ren.zoom(zoom)
            view_number = 1
            set_img_in_cell(mosaic, ren, view_number,
                            output_paths[view_number], width, height, i)

            ren.rm(slice_actor)
            slice_actor2 = slice_actor.copy()
            slice_actor2.display(None, slice_actor2.shape[1]//2, None)
            slice_actor2.opacity(opacity)
            ren.add(slice_actor2)

            ren.pitch(90)
            ren.set_camera(view_up=(0, 0, 1))
            ren.reset_camera()
            ren.zoom(zoom)
            view_number = 2
            set_img_in_cell(mosaic, ren, view_number,
                            output_paths[view_number], width, height, i)

            ren.pitch(180)
            ren.set_camera(view_up=(0, 0, 1))
            ren.reset_camera()
            ren.zoom(zoom)
            view_number = 3
            set_img_in_cell(mosaic, ren, view_number,
                            output_paths[view_number], width, height, i)

            ren.rm(slice_actor2)
            slice_actor3 = slice_actor.copy()
            slice_actor3.display(slice_actor3.shape[0]//2, None, None)
            slice_actor3.opacity(opacity)
            ren.add(slice_actor3)

            ren.yaw(90)
            ren.reset_camera()
            ren.zoom(zoom)
            view_number = 4
            set_img_in_cell(mosaic, ren, view_number,
                            output_paths[view_number], width, height, i)

            ren.yaw(180)
            ren.reset_camera()
            ren.zoom(zoom)
            view_number = 5
            set_img_in_cell(mosaic, ren, view_number,
                            output_paths[view_number], width, height, i)

            view_number = 6
            j = height * view_number
            draw_bundle_information(draw, bundle_file_name, number_streamlines,
                                    i + text_pos_x, j + text_pos_y, font)

        shutil.rmtree(output_bundle_dir)

    # Save image to file
    mosaic.save(args.output_name)


if __name__ == '__main__':
    main()
