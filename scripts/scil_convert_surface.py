#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert a surface (FreeSurfer or VTK supported).
    ".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"

> scil_convert_surface.py surf.vtk converted_surf.ply
"""
import argparse
import os
import vtk
from nibabel.freesurfer.io import read_geometry

from trimeshpy.vtk_util import (load_polydata,
                                save_polydata)

from scilpy.utils.util import (flip_LPS,
                               extract_xform)

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
                   help='Input a surface (FreeSurfer or supported by VTK).')

    p.add_argument('out_surface',
                   help='Output flipped surface (formats supported by VTK).')

    p.add_argument('--xform',
                   help='Path of the copy-paste output from mri_info \n'
                        'Using: mri_info $input >> log.txt, \n'
                        'The file log.txt would be this parameter')

    p.add_argument('--to_lps', action='store_true',
                   help='Flip for Surface/MI-Brain LPS')

    add_overwrite_arg(p)

    return p


def convert_with_vtk_legacy(surface_to_vtk, xform):

    surface = read_geometry(surface_to_vtk)
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()

    flip_LPS = [-1, -1, 1]

    for vertex in surface[0]:
        id = points.InsertNextPoint((vertex[0:3]+xform)*flip_LPS)

    for vertex_id in surface[1]:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, vertex_id[0])
        triangle.GetPointIds().SetId(1, vertex_id[1])
        triangle.GetPointIds().SetId(2, vertex_id[2])
        triangles.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    polydata.Modified()

    return polydata


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_surface)
    assert_outputs_exist(parser, args, args.out_surface)

    if args.xform:
        xform_matrix = extract_xform(args.xform)
        xform_translation = xform_matrix[0:3, 3]
    else:
        xform_translation = [0, 0, 0]

    if not ((os.path.splitext(args.in_surface)[1])
            in ['.vtk', '.vtp', '.fib', '.ply', '.stl', '.xml', '.obj']):
        polydata = convert_with_vtk_legacy(args.in_surface, xform_translation)

    else:
        polydata = load_polydata(args.out_surface)

    if args.to_lps:
        polydata = flip_LPS(polydata)

    save_polydata(polydata, args.out_surface, legacy_vtk_format=True)


if __name__ == "__main__":
    main()
