#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to flip a given surface (FreeSurfer or VTK supported).

Can flip vertices coordinates around a chosen (or multiple) axes (x, y or z)
as well as reverse the orientation of the surface normals.
"""

import argparse

from trimeshpy.io import load_mesh_from_file

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.surfaces.surface_operations import flip

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_surface',
                   help='Input surface (.vtk).')

    p.add_argument('out_surface',
                   help='Output flipped surface (.vtk).')

    p.add_argument('axes',
                   choices=['x', 'y', 'z', 'n'], nargs='+',
                   help='The axes you want to flip.'
                        ' eg: to flip the x and y axes use: x y.'
                        ' to reverse the surface normals use: n')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_surface)
    assert_outputs_exist(parser, args, args.out_surface)

    # Load mesh
    mesh = load_mesh_from_file(args.in_surface)

    mesh = flip(mesh, args.axes)
    # Save
    mesh.save(args.out_surface)


if __name__ == "__main__":
    main()
