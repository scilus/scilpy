#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 2-dimensional Bingham volume slice loaded from disk. The volume is
assumed to be saved from scil_compute_bingham_fodf_metrics.py.

Given an image of Bingham coefficients, this script displays a slice in a
given orientation.
"""

import argparse

import nibabel as nib
import numpy as np

from dipy.data import get_sphere, SPHERE_FILES

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.viz.scene_utils import (create_bingham_slicer,
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

    p.add_argument('--sphere', default='symmetric362',
                   choices=sorted(SPHERE_FILES.keys()),
                   help='Name of the sphere used to reconstruct SF. '
                        '[%(default)s]')

    p.add_argument('--color_per_lobe', action='store_true',
                   help='Color each bingham distribution with a '
                        'different color. [%(default)s]')

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
                                   args.slice_index, sph,
                                   args.color_per_lobe)

    # Prepare and display the scene
    scene = create_scene(actors, args.axis_name,
                         args.slice_index,
                         data.shape[:3])
    render_scene(scene, args.win_dims, args.interactor,
                 args.output, args.silent)


if __name__ == '__main__':
    main()
