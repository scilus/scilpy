#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to flip and reverse cortical surface (vtk or freesurfer).

Best usage for FreeSurfer to LPS vtk (for MI-Brain):
!!! important FreeSurfer surfaces must be in their respective folder !!!
> mris_convert --to-scanner lh.white lh.white.vtk
> scil_flip_surface.py lh.white.vtk lh_white_lps.vtk -x -y
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


def _build_args_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('surface',
                   help='Input surface (FreeSurfer or supported by VTK).')

    p.add_argument('out_surface',
                   help='Output flipped surface (formats supported by VTK).')

    p.add_argument('-x', action='store_true',
                   help='If supplied, flip the x axis.')
    p.add_argument('-y', action='store_true',
                   help='If supplied, flip the y axis.')
    p.add_argument('-z', action='store_true',
                   help='If supplied, flip the z axis.')

    p.add_argument('-n', '--inverse_normal', action='store_true',
                   help='If supplied, inverse surface orientation.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.surface)
    assert_outputs_exist(parser, args, args.out_surface)

    if not args.x and not args.y and not args.z and not args.inverse_normal:
        parser.error('No action specified (flipping axis or inverse normal).')

    # Load mesh
    mesh = load_mesh_from_file(args.surface)

    # Flip axes
    flip = [-1 if args.x else 1,
            -1 if args.y else 1,
            -1 if args.z else 1]
    tris, vts = mesh.flip_triangle_and_vertices(flip)
    mesh.set_vertices(vts)
    mesh.set_triangles(tris)

    # Reverse surface orientation
    if args.inverse_normal:
        tris = mesh.triangles_face_flip()
        mesh.set_triangles(tris)

    # Save
    mesh.save(args.out_surface)


if __name__ == "__main__":
    main()
