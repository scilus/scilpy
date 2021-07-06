#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 2-dimensional fODF slice loaded from disk.

Given an image of SH coefficients, this script displays a slice in a
given orientation. The user can also add a background on top of which the
fODF are to be displayed. Using a full SH basis, the script can be used to
visualize asymmetric fODF. The user can supply a peaks image to visualize
peaks on top of fODF.
"""

import argparse
import warnings

import nibabel as nib
import numpy as np

from dipy.data import get_sphere

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.viz.scene_utils import (create_bingham_slicer,
                                    create_texture_slicer,
                                    create_scene, render_scene)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    # Positional arguments
    p.add_argument('in_bingham', help='Input SH image file.')

    # Window configuration options
    p.add_argument('--slice_index', type=int,
                   help='Index of the slice to visualize along a given axis. '
                        'Defaults to middle of volume.')

    p.add_argument('--win_dims', nargs=2, metavar=('WIDTH', 'HEIGHT'),
                   default=(768, 768), type=int,
                   help='The dimensions for the vtk window. [%(default)s]')

    p.add_argument('--interactor', default='trackball',
                   choices={'image', 'trackball'},
                   help='Specify interactor mode for vtk window. '
                        '[%(default)s]')

    p.add_argument('--axis_name', default='axial', type=str,
                   choices={'axial', 'coronal', 'sagittal'},
                   help='Name of the axis to visualize. [%(default)s]')

    p.add_argument('--silent', action='store_true',
                   help='Disable interactive visualization.')

    p.add_argument('--output', help='Path to output file.')

    add_overwrite_arg(p)

    sphere_choices = {'symmetric362', 'symmetric642', 'symmetric724',
                      'repulsion724', 'repulsion100', 'repulsion200'}
    p.add_argument('--sphere', default='symmetric362', choices=sphere_choices,
                   help='Name of the sphere used to reconstruct SF. '
                        '[%(default)s]')

    p.add_argument('--sph_subdivide', type=int,
                   help='Number of subdivisions for given sphere. If not '
                        'supplied, use the given sphere as is.')

    p.add_argument('--colormap', default=None,
                   help='Colormap for the ODF slicer. If None, '
                        'then a RGB colormap will be used. [%(default)s]')

    p.add_argument('--scale', default=0.5, type=float,
                   help='Scaling factor for FODF. [%(default)s]')

    p.add_argument('--radial_scale_off', action='store_true',
                   help='Disable radial scale for ODF slicer.')

    p.add_argument('--norm_off', action='store_true',
                   help='Disable normalization of ODF slicer.')

    return p


def _parse_args(parser):
    args = parser.parse_args()
    inputs = []
    output = []
    inputs.append(args.in_bingham)
    if args.output:
        output.append(args.output)
    else:
        if args.silent:
            parser.error('Silent mode is enabled but no output is specified.'
                         'Specify an output with --output to use silent mode.')

    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, output)

    return args


def _get_data_from_inputs(args):
    """
    Load data given by args. Perform checks to ensure dimensions agree
    between the data for mask, background, peaks and fODF.
    """
    bingham = nib.nifti1.load(args.in_bingham).get_fdata(dtype=np.float32)
    return bingham


def main():
    parser = _build_arg_parser()
    args = _parse_args(parser)
    data = _get_data_from_inputs(args)
    sph = get_sphere(args.sphere)

    actors = create_bingham_slicer(data, args.axis_name,
                                   args.slice_index, sph)

    # Prepare and display the scene
    scene = create_scene(actors, args.axis_name,
                         args.slice_index,
                         data.shape[:3])
    render_scene(scene, args.win_dims, args.interactor,
                 args.output, args.silent)


if __name__ == '__main__':
    main()
