#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import nibabel as nib

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)

from dipy.viz import actor, window
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Visualize bundles from a list. '
                    'The script will output one screenshot per direction '
                    '(6 total) for each bundle in the list.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('anat_reference',
                        help='Image used as background (e.g. T1, FA, b0).')
    parser.add_argument('inputs', nargs='+',
                        help='List of streamline files supported by nibabel.')
    parser.add_argument('output_name',
                        help='Name of the output image mosaic (e.g. jpg, png).')
    parser.add_argument('--zoom', type=float, default=1.0,
                        help='Rendering zoom. '
                        'A value greater than 1 is a zoom-in, a value less than 1 is a zoom-out.')
    parser.add_argument('--ttf', default=None,
                        help='Path of the true type font to use for the legends.')
    parser.add_argument('--ttf_size', type=int, default=35,
                        help='Font size (int) to use for the legends.')
    parser.add_argument('--opacity_background', type=float, default=0.4,
                        help='Opacity of background image [0, 1.0]')

    add_overwrite_arg(parser)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.anat_reference])
    assert_outputs_exists(parser, args, [args.output_name])

    output_names = ['axial_superior', 'axial_inferior',
                    'coronal_posterior', 'coronal_anterior',
                    'sagittal_left', 'sagittal_right']

    listOfBundles = [f for f in args.inputs]
    output_dir = os.path.dirname(args.output_name)  # Where temporal files will be created
    # ----------------------------------------------------------------------- #
    # Mosaic, column 0: orientation names and data description
    # ----------------------------------------------------------------------- #
    width = 300
    height = 300
    rows = 6
    cols = len(listOfBundles)
    text_pos_x = 50
    text_pos_y = 50

    # Creates a new empty image, RGB mode
    mosaic = Image.new('RGB', ((cols + 1) * width, (rows + 1) * height))
    # Prepare draw and font objects to render text
    draw = ImageDraw.Draw(mosaic)
    if args.ttf is not None:
        try:
            font = ImageFont.truetype(args.ttf, args.ttf_size)
        except:
            print('Font %s was not found. Default font will be used.' % (args.ttf))
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # T1 as background
    ref_img = nib.load(args.anat_reference)
    data = ref_img.get_data()
    affine = ref_img.affine
    mean, std = data[data > 0].mean(), data[data > 0].std()
    value_range = (mean - 0.5 * std, mean + 1.5 * std)

    stats = {}
    # Orientation's names
    for num, name in enumerate(output_names):
        i = 0
        j = height * num + 50
        # Name splited in two lines
        draw.text((i + text_pos_x, j + text_pos_y),
                  name[:name.find('_')], font=font)
        draw.text((i + text_pos_x, j + text_pos_y + 50),
                  name[1+name.find('_'):], font=font)

    # First column, last row: description of the information to show
    viewNumber = 6
    i = 0
    j = height * viewNumber
    draw.text((i + text_pos_x, j + text_pos_y),
              ('Bundle'), font=font)
    draw.text((i + text_pos_x, j + text_pos_y + 50),
              ('Streamlines'), font=font)

    # ----------------------------------------------------------------------- #
    # Columns with bundles
    # ----------------------------------------------------------------------- #
    for numBundle, bundle_file in enumerate(listOfBundles):

        bundle_file_name = os.path.basename(bundle_file)
        bundle_name, _ = os.path.splitext(bundle_file_name)

        # !! It creates a temporal folder to create the images to concatenate in the mosaic !!
        output_bundle_dir = os.path.join(output_dir, bundle_name)
        if not os.path.isdir(output_bundle_dir):
            os.makedirs(output_bundle_dir)

        output_paths = [
            os.path.join(output_bundle_dir,
                         '{}_' + os.path.basename(output_bundle_dir)).format(name)
            for name in output_names]
        stats[bundle_name] = {}

        i = (numBundle + 1)*width

        if not os.path.isfile(bundle_file):
            print('\nInput file %s doesn\'t exist.' % (bundle_file))

            viewNumber = 6
            j = height * viewNumber

            stats[bundle_name]['number_streamlines'] = 0

            draw.text((i + text_pos_x, j + text_pos_y),
                      (bundle_file_name),
                      font=font)
            draw.text((i + text_pos_x, j + text_pos_y + 50),
                      ('%d' % stats[bundle_name]['number_streamlines']),
                      font=font)

        else:
            # Select the streamlines to plot
            bundle_tractogram_file = nib.streamlines.load(bundle_file)
            streamlines = bundle_tractogram_file.streamlines

            tubes = actor.line(streamlines)

            # Render
            ren = window.Renderer()
            zoom = args.zoom
            opacity = args.opacity_background

            # Structural data
            slice_actor = actor.slicer(data, affine, value_range)
            slice_actor.opacity(opacity)
            ren.add(slice_actor)

            slice_actor2 = slice_actor.copy()
            slice_actor2.display(slice_actor2.shape[0]//2, None, None)
            slice_actor2.opacity(opacity)
            ren.add(slice_actor2)

            slice_actor3 = slice_actor.copy()
            slice_actor3.display(None, slice_actor3.shape[1]//2, None)
            slice_actor3.opacity(opacity)
            ren.add(slice_actor3)

            ren.add(tubes)
            ren.reset_camera()
            ren.zoom(zoom)
            window.snapshot(ren, output_paths[0])
            viewNumber = 0
            j = height * viewNumber
            image = Image.open(output_paths[viewNumber])
            image.thumbnail((width, height))
            mosaic.paste(image, (i, j))

            ren.pitch(180)
            ren.reset_camera()
            ren.zoom(zoom)
            window.snapshot(ren, output_paths[1])
            viewNumber = 1
            j = height * viewNumber
            image = Image.open(output_paths[viewNumber])
            image.thumbnail((width, height))
            mosaic.paste(image, (i, j))

            ren.pitch(90)
            ren.set_camera(view_up=(0, 0, 1))
            ren.reset_camera()
            ren.zoom(zoom)
            window.snapshot(ren, output_paths[2])
            viewNumber = 2
            j = height * viewNumber
            image = Image.open(output_paths[viewNumber])
            image.thumbnail((width, height))
            mosaic.paste(image, (i, j))

            ren.pitch(180)
            ren.set_camera(view_up=(0, 0, 1))
            ren.reset_camera()
            ren.zoom(zoom)
            window.snapshot(ren, output_paths[3])
            viewNumber = 3
            j = height * viewNumber
            image = Image.open(output_paths[viewNumber])
            image.thumbnail((width, height))
            mosaic.paste(image, (i, j))

            ren.yaw(90)
            ren.reset_camera()
            ren.zoom(zoom)
            window.snapshot(ren, output_paths[4])
            viewNumber = 4
            j = height * viewNumber
            image = Image.open(output_paths[viewNumber])
            image.thumbnail((width, height))
            mosaic.paste(image, (i, j))

            ren.yaw(180)
            ren.reset_camera()
            ren.zoom(zoom)
            window.snapshot(ren, output_paths[5])
            viewNumber = 5
            j = height * viewNumber
            image = Image.open(output_paths[viewNumber])
            image.thumbnail((width, height))
            mosaic.paste(image, (i, j))

            viewNumber = 6
            j = height * viewNumber

            stats[bundle_name]['number_streamlines'] = len(streamlines)

            draw.text((i + text_pos_x, j + text_pos_y),
                      (bundle_file_name), font=font)
            draw.text((i + text_pos_x, j + text_pos_y + 50),
                      ('%d' % stats[bundle_name]['number_streamlines']), font=font)

        shutil.rmtree(output_bundle_dir)

    # Save image to file
    mosaic.save(args.output_name)


if __name__ == '__main__':
    main()
