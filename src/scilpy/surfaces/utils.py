# -*- coding: utf-8 -*-

import numpy as np
import vtk
import nibabel as nib
from nibabel.freesurfer.io import read_geometry


def convert_freesurfer_into_polydata(surface_to_polydata, reference):
    """
    Convert a freesurfer surface into a polydata surface with vtk.

    Parameters
    ----------
    surface_to_vtk: Input a surface from freesurfer.
        The header must not contain any of these suffixes:
        '.vtk', '.vtp', '.fib', '.ply', '.stl', '.xml', '.obj'.

    reference: Reference image to extract the transformation matrix.
        The reference image is used to extract the transformation matrix

    Returns
    -------
    polydata : A polydata surface.
        A polydata is a mesh structure that can hold data arrays
        in points, cells, or in the dataset itself.
    """
    surface = read_geometry(surface_to_polydata)
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()

    img = nib.load(reference)
    affine = img.affine
    center_volume = (np.array(img.shape) / 2)

    xform_translation = np.dot(affine[0:3, 0:3], center_volume) + affine[0:3, 3]

    for vertex in surface[0]:
        _ = points.InsertNextPoint((vertex[0:3] + xform_translation))

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


def flip_surfaces_axes(polydata, axes=[-1, -1, 1]):
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
    flip_matrix = vtk.vtkMatrix4x4()
    flip_matrix.Identity()

    for i in range(3):
        flip_matrix.SetElement(i, i, axes[i])

    # Apply the transforms
    transform = vtk.vtkTransform()
    transform.Concatenate(flip_matrix)

    # Transform the polydata
    transform_polydata = vtk.vtkTransformPolyDataFilter()
    transform_polydata.SetTransform(transform)
    transform_polydata.SetInputData(polydata)
    transform_polydata.Update()
    polydata = transform_polydata.GetOutput()

    return polydata
