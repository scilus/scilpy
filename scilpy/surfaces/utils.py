# -*- coding: utf-8 -*-

import numpy as np
import vtk
from nibabel.freesurfer.io import read_geometry


def convert_freesurfer_into_polydata(surface_to_polydata, xform):
    """
    Convert a freesurfer surface into a polydata surface with vtk.

    Parameters
    ----------
    surface_to_vtk: Input a surface from freesurfer.
        The header must not contain any of these suffixes:
        '.vtk', '.vtp', '.fib', '.ply', '.stl', '.xml', '.obj'.

    xform: array [float]
        Apply a transformation matrix to the surface to align
        freesurfer surface with T1.

    Returns
    -------
    polydata : A polydata surface.
        A polydata is a mesh structure that can hold data arrays
        in points, cells, or in the dataset itself.
    """
    surface = read_geometry(surface_to_polydata)
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


def extract_xform(xform):
    """
    Use the log.txt file from mri_info to generate a transformation
    matrix to align the freesurfer surface with the T1.

    Parameters
    ----------
    filename : list
        The copy-paste output from mri_info of the surface using:
        mri_info $surface >> log.txt

    Returns
    -------
    Matrix : np.array
        a transformation matrix to align the surface with the T1.
    """

    raw_xform = []
    for i in xform:
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
    """
    Apply a flip to the freesurfer surface of the anteroposterior axis.

    Parameters
    ----------
    polydata : polydata surface.
        A surface mesh structure after a transformation in polydata
        surface with vtk.

    Returns
    -------
    polydata : polydata surface.
        return the polydata turned over.
    """
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
