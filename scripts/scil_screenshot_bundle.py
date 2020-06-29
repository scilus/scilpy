#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Register bundle to a template for screenshots using a reference.
The template can be any MNI152 (any resolution, cropped or not)
If your in_anat has a skull, select a MNI152 template with a skull and
vice-versa.

If the bundle is already in MNI152 space, do not use --target_template.

Axial, coronal and sagittal slices are captured.
Sagittal can be capture from the left (default) or the right.
"""

import argparse
import logging
import os

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import transform_streamlines
from fury import actor
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
import numpy as np
from scipy.ndimage import map_coordinates

from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.image import register_image
from scilpy.viz.screenshot import display_slices


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Path of the input bundle.')
    p.add_argument('in_anat',
                   help='Path of the reference file (.nii or nii.gz).')
    p.add_argument('--target_template',
                   help='Path to the target MNI152 template for registration. \n'
                        'If in_anat has a skull, select a MNI152 template \n'
                        'with a skull and vice-versa.')

    sub_color = p.add_mutually_exclusive_group()
    sub_color.add_argument('--local_coloring', action='store_true',
                           help='Color streamlines local segments orientation.')
    sub_color.add_argument('--uniform_coloring', nargs=3,
                           metavar=('R', 'G', 'B'), type=float,
                           help='Color streamlines with uniform coloring.')
    sub_color.add_argument('--reference_coloring',
                           metavar='COLORBAR',
                           help='Color streamlines with reference coloring '
                                '(0-255).')

    p.add_argument('--right', action='store_true',
                   help='Take screenshot from the right instead of the left \n'
                        'for the sagittal plane.')
    p.add_argument('--anat_opacity', type=float, default=0.3,
                   help='Set the opacity for the anatomy, use 0 for complete \n'
                        'transparency, 1 for opaque. [%(default)s]')
    p.add_argument('--output_suffix',
                   help='Add a suffix to the output, else the axis name is used.')
    p.add_argument('--out_dir', default='',
                   help='Put all images in a specific directory.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def prepare_data_for_actors(bundle_filename, reference_filename,
                            target_template_filename):
    sft = load_tractogram(bundle_filename, reference_filename)
    streamlines = sft.streamlines

    # Load and prepare the data
    reference_img = nib.load(reference_filename)
    reference_data = reference_img.get_fdata(dtype=np.float32)
    reference_affine = reference_img.affine

    if target_template_filename:
        target_template_img = nib.load(target_template_filename)
        target_template_data = target_template_img.get_fdata(dtype=np.float32)
        target_template_affine = target_template_img.affine

        # Register the DWI data to the template
        logging.debug('Starting registration...')
        transformed_reference, transformation = register_image(target_template_data,
                                                               target_template_affine,
                                                               reference_data,
                                                               reference_affine)
        logging.debug('Transforming streamlines...')
        streamlines = transform_streamlines(streamlines,
                                            np.linalg.inv(transformation),
                                            in_place=True)

        new_sft = StatefulTractogram(streamlines, target_template_filename,
                                     Space.RASMM)

        return new_sft, transformed_reference

    return sft, reference_data


def plot_glass_brain(args, sft, img, output_filenames):
    sft.to_vox()
    sft.to_corner()
    _, dimensions, _, _ = sft.space_attributes
    data = compute_tract_counts_map(sft.streamlines, dimensions)
    data[data > 100] = 100
    img = nib.Nifti1Image(data, img.affine)

    axes = 'yz'
    if args.right:
        axes = 'r' + axes
    else:
        axes = 'l' + axes

    for i, axe in enumerate(axes):
        display = plotting.plot_glass_brain(img,
                                            black_bg=True,
                                            display_mode=axe,
                                            alpha=0.5)
        display.savefig(output_filenames[i], dpi=300)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    required = [args.in_bundle, args.in_anat]
    assert_inputs_exist(parser, required, args.target_template)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    output_filenames_3d = []
    output_filenames_glass = []
    for axis_name in ['sagittal', 'coronal', 'axial']:
        if args.output_suffix:
            output_filenames_3d.append(os.path.join(args.out_dir,
                                                    '{0}_{1}_3d.png'.format(
                                                        axis_name,
                                                        args.output_suffix)))

            output_filenames_glass.append(os.path.join(args.out_dir,
                                                       '{0}_{1}_glass.png'.format(
                                                           axis_name,
                                                           args.output_suffix)))
        else:
            output_filenames_3d.append(os.path.join(args.out_dir,
                                                    '{0}_3d.png'.format(
                                                        axis_name)))
            output_filenames_glass.append(os.path.join(args.out_dir,
                                                       '{0}_glass.png'.format(
                                                           axis_name)))

    assert_outputs_exist(parser, args,
                         output_filenames_3d+output_filenames_glass)

    if args.out_dir and not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if args.anat_opacity < 0.0 or args.anat_opacity > 1.0:
        parser.error('Opacity must be between 0 and 1')

    if args.uniform_coloring:
        for val in args.uniform_coloring:
            if val < 0 or val > 255:
                parser.error('{0} is not a valid RGB value'.format(val))

    # Get the relevant slices from the template
    if args.target_template:
        mni_space_img = nib.load(args.target_template)
        affine = nib.load(args.target_template).affine
    else:
        mni_space_img = nib.load(args.in_anat)
        affine = nib.load(args.in_anat).affine

    x_slice = int(mni_space_img.shape[0] / 2)
    y_slice = int(mni_space_img.shape[1] / 2)
    z_slice = int(mni_space_img.shape[2] / 2)
    slices_choice = (x_slice, y_slice, z_slice)

    subject_data = prepare_data_for_actors(args.in_bundle, args.in_anat,
                                           args.target_template)

    # Create actors from each dataset for Dipy
    sft, reference_data = subject_data
    streamlines = sft.streamlines

    volume_actor = actor.slicer(reference_data,
                                affine=affine,
                                opacity=args.anat_opacity,
                                interpolation='nearest')
    if args.local_coloring:
        colors = []
        for i in streamlines:
            local_color = np.gradient(i, axis=0)
            local_color = np.abs(local_color)
            local_color = (local_color.T / np.max(local_color, axis=1)).T
            colors.append(local_color)
    elif args.uniform_coloring:
        colors = (args.uniform_coloring[0] / 255.0,
                  args.uniform_coloring[1] / 255.0,
                  args.uniform_coloring[2] / 255.0)
    elif args.reference_coloring:
        sft.to_vox()
        streamlines_vox = sft.get_streamlines_copy()
        sft.to_rasmm()
        colors = []
        normalized_data = reference_data / np.max(reference_data)
        cmap = plt.get_cmap(args.reference_coloring)
        for points in streamlines_vox:
            values = map_coordinates(normalized_data, points.T,
                                     order=1, mode='nearest')
            colors.append(cmap(values)[:, 0:3])
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
                   output_filenames_3d[0], 'sagittal',
                   view_position=tuple([x for x in side_pos]),
                   focal_point=tuple([x for x in (0, -10, 10)]),
                   streamlines_actor=streamlines_actor)
    display_slices(volume_actor, slices_choice,
                   output_filenames_3d[1], 'coronal',
                   view_position=tuple([x for x in (0, -300, 15)]),
                   focal_point=tuple([x for x in (0, 0, 15)]),
                   streamlines_actor=streamlines_actor)
    display_slices(volume_actor, slices_choice,
                   output_filenames_3d[2], 'axial',
                   view_position=tuple([x for x in (0, -15, 350)]),
                   focal_point=tuple([x for x in (0, -15, 0)]),
                   streamlines_actor=streamlines_actor)

    plot_glass_brain(args, sft, mni_space_img, output_filenames_glass)


if __name__ == "__main__":
    main()
