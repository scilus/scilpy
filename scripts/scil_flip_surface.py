#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to flip and reverse a surface (FreeSurfer or VTK supported).
Can be used to flip in chosen axes (x, y or z),
it can also flip inside out the surface orientation (normal).


Best usage for FreeSurfer to LPS vtk (for MI-Brain):
!!! important FreeSurfer surfaces must be in their respective folder !!!
> mris_convert --to-scanner lh.white lh.white.vtk
> scil_flip_surface.py lh.white.vtk lh_white_lps.vtk x y
"""

import argparse

from trimeshpy.io import load_mesh_from_file

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)

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
                   help='The axes (or normal orientation) you want to flip.'
                        ' eg: to flip the x and y axes use: x y.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_surface)
    assert_outputs_exist(parser, args, args.out_surface)

    # Load mesh
    mesh = load_mesh_from_file(args.in_surface)

    # Flip axes
    flip = (-1 if 'x' in args.axes else 1,
            -1 if 'y' in args.axes else 1,
            -1 if 'z' in args.axes else 1)
    tris, vts = mesh.flip_triangle_and_vertices(flip)
    mesh.set_vertices(vts)
    mesh.set_triangles(tris)

    # Reverse surface orientation
    if 'n' in args.axes:
        tris = mesh.triangles_face_flip()
        mesh.set_triangles(tris)

    # Save
    mesh.save(args.out_surface)


if __name__ == "__main__":
    main()
