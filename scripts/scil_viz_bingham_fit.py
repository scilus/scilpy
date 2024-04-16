#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 2-dimensional Bingham volume slice loaded from disk. The volume is
assumed to be saved from scil_fodf_to_bingham.py.

Given an image of Bingham coefficients, this script displays a slice in a
given orientation.
"""

import argparse
import logging

import nibabel as nib

from dipy.data import get_sphere, SPHERE_FILES

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.spatial import RAS_AXES_NAMES
from scilpy.utils.spatial import get_axis_index

from scilpy.viz.backends.fury import (create_interactive_window,
                                      create_scene,
                                      snapshot_scenes)
from scilpy.viz.screenshot import compose_image
from scilpy.viz.slice import create_bingham_slicer


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
                   choices=RAS_AXES_NAMES,
                   help='Name of the axis to visualize. [%(default)s]')

    p.add_argument('--silent', action='store_true',
                   help='Disable interactive visualization.')

    p.add_argument('--output', help='Path to output file.')

    add_verbose_arg(p)
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


def _get_slicing_for_axis(axis_name, index, shape):
    """
    Get a tuple of slice representing the slice of interest at `index`
    along the axis `axis_name` in an input volume of dimensions `shape`.
    """
    slicing = [slice(shape[0]), slice(shape[1]), slice(shape[2])]
    slicing[get_axis_index(axis_name)] = slice(index, index + 1)
    return tuple(slicing)


def _get_data_from_inputs(args):
    """
    Load data given by args.
    """
    bingham = nib.load(args.in_bingham).get_fdata()
    if not args.slice_index:
        slice_index = bingham.shape[get_axis_index(args.axis_name)] // 2
    else:
        slice_index = args.slice_index
    bingham = bingham[_get_slicing_for_axis(args.axis_name,
                                            slice_index,
                                            bingham.shape)]
    return bingham


def main():
    parser = _build_arg_parser()
    args = _parse_args(parser)
    data = _get_data_from_inputs(args)
    sph = get_sphere(args.sphere)
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    actors = create_bingham_slicer(data, args.axis_name,
                                   args.slice_index, sph,
                                   args.color_per_lobe)

    # Prepare and display the scene
    scene = create_scene(actors, args.axis_name,
                         args.slice_index,
                         data.shape[:3],
                         args.win_dims[0] / args.win_dims[1])

    if not args.silent:
        create_interactive_window(
            scene, args.win_dims, args.interactor)

    if args.output:
        # Legacy. When this snapshotting gets updated to align with the
        # viz module, snapshot_scenes should be called directly
        snapshot = next(snapshot_scenes([scene], args.win_dims))
        image = compose_image(snapshot, args.win_dims, args.slice_index)
        image.save(args.output)


if __name__ == '__main__':
    main()
