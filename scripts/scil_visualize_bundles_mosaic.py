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
import shutil

from dipy.io.utils import is_header_compatible
from fury import actor, window
import nibabel as nib
import numpy as np

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.utils.filenames import split_name_with_nii


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

    add_overwrite_arg(p)
    return p


def get_font(args):
    """ Returns a ttf font object. """
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
              ('Elements'), font=font)


def draw_bundle_information(draw, bundle_file_name, nbr_of_elem,
                            pos_x, pos_y, font):
    """ Draw text with bundle information. """
    draw.text((pos_x, pos_y),
              (bundle_file_name), font=font)
    draw.text((pos_x, pos_y + font.getsize(' ')[1]*1.5),
              ('{}'.format(nbr_of_elem)), font=font)


def set_img_in_cell(mosaic, ren, view_number, path, width, height, i):
    """ Set a snapshot of the bundle in a cell of mosaic """
    window.snapshot(ren, path, size=(width, height))
    j = height * view_number
    image = Image.open(path)
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

    assert_inputs_exist(parser, args.in_volume)
    assert_outputs_exist(parser, args, args.out_image)

    output_names = ['axial_superior', 'axial_inferior',
                    'coronal_posterior', 'coronal_anterior',
                    'sagittal_left', 'sagittal_right']

    for filename in args.in_bundles:
        if not is_header_compatible(args.in_volume, filename):
            parser.error('{} does not have a compatible header with {}'.format(
                filename, args.in_volume))

    output_dir = os.path.dirname(args.out_image)
    if output_dir:
        assert_output_dirs_exist_and_empty(parser, args, output_dir,
                                           create_dir=True)

    # ----------------------------------------------------------------------- #
    # Mosaic, column 0: orientation names and data description
    # ----------------------------------------------------------------------- #
    width = args.resolution_of_thumbnails
    height = args.resolution_of_thumbnails
    rows = 6
    cols = len(args.in_bundles)
    text_pos_x = 50
    text_pos_y = 50

    # Creates a new empty image, RGB mode
    mosaic = Image.new('RGB', ((cols + 1) * width, (rows + 1) * height))

    # Prepare draw and font objects to render text
    draw = ImageDraw.Draw(mosaic)
    font = get_font(args)

    # Data of the volume used as background
    ref_img = nib.load(args.in_volume)
    data = ref_img.get_fdata(dtype=np.float32)
    affine = ref_img.affine
    mean, std = data[data > 0].mean(), data[data > 0].std()
    value_range = (mean - 0.5 * std, mean + 1.5 * std)

    # First column with rows description
    draw_column_with_names(draw, output_names, text_pos_x,
                           text_pos_y, height, font)

    # ----------------------------------------------------------------------- #
    # Columns with bundles
    # ----------------------------------------------------------------------- #
    random.seed(args.random_coloring)
    for idx_bundle, bundle_file in enumerate(args.in_bundles):

        bundle_file_name = os.path.basename(bundle_file)
        bundle_name, bundle_ext = split_name_with_nii(bundle_file_name)

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
            if args.uniform_coloring:
                colors = args.uniform_coloring
            elif args.random_coloring is not None:
                colors = random_rgb()
            # Select the streamlines to plot
            if bundle_ext in ['.tck', '.trk']:
                if args.random_coloring is None and args.uniform_coloring is None:
                    colors = None
                bundle_tractogram_file = nib.streamlines.load(bundle_file)
                streamlines = bundle_tractogram_file.streamlines
                bundle_actor = actor.line(streamlines, colors)
                nbr_of_elem = len(streamlines)
            # Select the volume to plot
            elif bundle_ext in ['.nii.gz', '.nii']:
                if not args.random_coloring and not args.uniform_coloring:
                    colors = [1.0, 1.0, 1.0]
                bundle_img_file = nib.load(bundle_file)
                roi = get_data_as_mask(bundle_img_file)
                bundle_actor = actor.contour_from_roi(roi,
                                                      bundle_img_file.affine,
                                                      colors)
                nbr_of_elem = np.count_nonzero(roi)

            # Render
            ren = window.Scene()
            zoom = args.zoom
            opacity = args.opacity_background

            # Structural data
            slice_actor = actor.slicer(data, affine, value_range)
            slice_actor.opacity(opacity)
            ren.add(slice_actor)

            # Streamlines
            ren.add(bundle_actor)
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
            draw_bundle_information(draw, bundle_file_name, nbr_of_elem,
                                    i + text_pos_x, j + text_pos_y, font)

        shutil.rmtree(output_bundle_dir)

    # Save image to file
    mosaic.save(args.out_image)


if __name__ == '__main__':
    main()
