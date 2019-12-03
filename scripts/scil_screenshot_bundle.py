#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os

from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import transform_streamlines
from fury import actor
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.image import register_image
from scilpy.viz.screenshot import display_slices

DESCRIPTION = """
   Register bundle to a template for screenshots using a reference.
   The template are in /mnt/braindata/Other/simple_template_viz/
   Axial, coronal and sagittal slices are captured.
   Sagittal can be capture from the left (default) or the right.
   """


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Path of the input bundle')
    p.add_argument('in_anat',
                   help='Path of the reference file (.nii or nii.gz)')
    p.add_argument('target_template',
                   help='Path to the target MNI152template for registration, '
                        'Any choice of modality works')

    sub_color = p.add_mutually_exclusive_group()
    sub_color.add_argument('--uniform_coloring', nargs=3, metavar=('R', 'G', 'B'),
                           type=int,
                           help='Color streamlines with uniform coloring')
    sub_color.add_argument('--reference_coloring', action='store_true',
                           help='Color streamlines with reference coloring (0-255)')

    p.add_argument('--right', action='store_true',
                   help='Take screenshot from the right instead of the left \n'
                        'for the sagittal plane')
    p.add_argument('--anat_opacity', type=float, default=0.3,
                   help='Set the opacity for the anatomy, use 0 for complete '
                   'transparency, 1 for opaque')
    p.add_argument('--output_suffix',
                   help='Add a suffix to the output, else the axis name is used')
    p.add_argument('--output_dir', default='',
                   help='Put all images in a specific directory, will overwrite')
    add_overwrite_arg(p)

    return p


def prepare_data_for_actors(bundle_filename, reference_filename,
                            target_template_filename):
    sft = load_tractogram(bundle_filename, reference_filename)
    # sft.to_vox()
    streamlines = sft.streamlines

    # Load and prepare the data
    reference_img = nib.load(reference_filename)
    reference_data = reference_img.get_data()
    reference_affine = reference_img.get_affine()

    target_template_img = nib.load(target_template_filename)
    target_template_data = target_template_img.get_data()
    target_template_affine = target_template_img.affine
    zooms = 1 / float(target_template_img.header.get_zooms()[0])

    # Register the DWI data to the template
    transformed_reference, transformation = register_image(target_template_data,
                                                           target_template_affine,
                                                           reference_data,
                                                           reference_affine)
    # transformation = np.eye(4)
    streamlines = transform_streamlines(streamlines,
                                        np.linalg.inv(transformation))
    # streamlines = transform_streamlines(streamlines, zooms * np.eye(4))

    return streamlines, transformed_reference


def main():
    parser = _build_args_parser()
    args = parser.parse_args()
    required = [args.in_bundle, args.in_anat, args.target_template]
    assert_inputs_exist(parser, required)

    output_filenames = []
    for axis_name in ['sagittal', 'coronal', 'axial']:
        if args.output_suffix:
            output_filenames.append(os.path.join(args.output_dir,
                                                 '{0}_{1}.png'.format(
                                                     axis_name,
                                                     args.output_suffix)))
        else:
            output_filenames.append(os.path.join(args.output_dir,
                                                 '{0}.png'.format(axis_name)))

    assert_outputs_exist(parser, args, output_filenames)

    if args.output_dir and not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

        if args.anat_opacity < 0.0 or args.anat_opacity > 1.0:
            parser.error('Opacity must be between 0 and 1')

    if args.uniform_coloring:
        for val in args.uniform_coloring:
            if val < 0 or val > 255:
                parser.error('{0} is not a valid RGB value'.format(val))

    # Get the relevant slices from the template
    target_template_img = nib.load(args.target_template)
    zooms = 1

    x_slice = int(target_template_img.shape[0] / 2)
    y_slice = int(target_template_img.shape[1] / 2)
    z_slice = int(target_template_img.shape[2] / 2)
    slices_choice = (x_slice, y_slice, z_slice)

    subject_data = prepare_data_for_actors(args.in_bundle, args.in_anat,
                                           args.target_template)

    # Create actors from each dataset for Dipy
    streamlines, reference_data = subject_data
    volume_actor = actor.slicer(reference_data,
                                affine=nib.load(args.target_template).affine,
                                opacity=args.anat_opacity,
                                interpolation='nearest')

    if args.uniform_coloring:
        colors = (args.uniform_coloring[0] / 255.0,
                  args.uniform_coloring[1] / 255.0,
                  args.uniform_coloring[2] / 255.0)
    elif args.reference_coloring:
        colors = reference
    else:
        colors = None

    streamlines_actor = actor.line(streamlines, colors=colors, linewidth=0.2)

    # Take a snapshot of each dataset, camera settings are fixed for the
    # known template, won't work with another.
    if args.right:
        side_pos = (300, -10, 10)
    else:
        side_pos = (-300, 10, 10)
    display_slices(volume_actor, slices_choice,
                   output_filenames[0], 'sagittal',
                   view_position=tuple([x for x in side_pos]),
                   focal_point=tuple([x for x in (0, -10, 10)]),
                   streamlines_actor=streamlines_actor)
    display_slices(volume_actor, slices_choice,
                   output_filenames[1], 'coronal',
                   view_position=tuple([x for x in (0, 250, 15)]),
                   focal_point=tuple([x for x in (0, 0, 15)]),
                   streamlines_actor=streamlines_actor)
    display_slices(volume_actor, slices_choice,
                   output_filenames[2], 'axial',
                   view_position=tuple([x for x in (0, -15, 350)]),
                   focal_point=tuple([x for x in (0, -15, 0)]),
                   streamlines_actor=streamlines_actor)


if __name__ == "__main__":
    main()
