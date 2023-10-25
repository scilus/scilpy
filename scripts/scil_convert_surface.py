#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert a surface (FreeSurfer or VTK supported).
    ".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"

> scil_convert_surface.py surf.vtk converted_surf.ply
"""
import argparse
import os

import nibabel as nib
from nibabel.freesurfer.io import read_geometry
import numpy as np
import vtk

from trimeshpy.io import load_mesh_from_file
from trimeshpy.vtk_util import save_polydata

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteau, M. 2018.
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
                   help='Flip for Surfice/MI-Brain LPS')
    
    add_overwrite_arg(p)

    return p

def extract_xform(filename):
    with open(filename) as f:
        content = f.readlines()
    names = [x.strip() for x in content]

    raw_xform = []
    for i in names:
        raw_xform.extend(i.split())

    start_read = 0
    for i, value in enumerate(raw_xform):
        if value == 'xform':
            start_read = int(i)
            break

    if start_read == 0:
        raise ValueError('No xform in that file...')

    matrix = np.eye(4)
    for i in range(3):
        for j in range(4):
            matrix[i, j] = float(raw_xform[13*i + (j*3) + 4+2+start_read][:-1])
    return matrix

def flip_LPS(polydata):
    flip_LPS = vtk.vtkMatrix4x4()
    flip_LPS.Identity()
    flip_LPS.SetElement(0, 0, -1)
    flip_LPS.SetElement(1, 1, -1)

    # Apply the transforms
    transform = vtk.vtkTransform()
    transform.Concatenate(flip_LPS)

    # Apply the transforms
    transform = vtk.vtkTransform()
    transform.Concatenate(flip_LPS)

    # Transform the polydata
    transform_polydata = vtk.vtkTransformPolyDataFilter()
    transform_polydata.SetTransform(transform)
    transform_polydata.SetInputData(polydata)
    transform_polydata.Update()
    polydata = transform_polydata.GetOutput()

    return polydata

def convert_with_vtk_legacy(surface_to_vtk):
    parser = _build_arg_parser()
    args = parser.parse_args()

    surface = read_geometry(surface_to_vtk)
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()

    flip_LPS = [-1, -1, 1]

    if args.xform:
        xform_matrix = extract_xform(args.xform)
        xform_translation = xform_matrix[0:3, 3]
    else:
        xform_translation = [0, 0, 0]

    # All possibles points in the VTK Polydata
    for vertex in surface[0]:
        id = points.InsertNextPoint((vertex[0:3]+xform_translation)*flip_LPS)

    # Each triangle is defined by an ID
    for vertex_id in surface[1]:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, vertex_id[0]);
        triangle.GetPointIds().SetId(1, vertex_id[1]);
        triangle.GetPointIds().SetId(2, vertex_id[2]);
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

    if not ((os.path.splitext(args.in_surface)[1])in['.vtk','.vtp','.fib','.ply','.stl','.xml','.obj']):
        polydata = convert_with_vtk_legacy(args.in_surface)

    else:
        mesh = load_mesh_from_file(args.in_surface)
        polydata = mesh.save(args.out_surface)

    if args.to_lps:
        polydata = flip_LPS(polydata)

    # Save the output and write it in .vtk
    save_polydata(polydata, args.out_surface, legacy_vtk_format=True)

if __name__ == "__main__":
    main()
