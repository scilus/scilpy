#! /usr/bin/env python
# -*- coding: utf-8 -*-

from fury import window
import numpy as np
import vtk
from vtk.util import numpy_support


def display_slices(volume_actor, slices,
                   output_filename, axis_name,
                   view_position, focal_point,
                   peaks_actor=None, streamlines_actor=None):
    # Setting for the slice of interest
    if axis_name == 'sagittal':
        volume_actor.display(slices[0], None, None)
        if peaks_actor:
            peaks_actor.display(slices[0], None, None)
        view_up_vector = (0, 0, 1)
    elif axis_name == 'coronal':
        volume_actor.display(None, slices[1], None)
        if peaks_actor:
            peaks_actor.display(None, slices[1], None)
        view_up_vector = (0, 0, 1)
    else:
        volume_actor.display(None, None, slices[2])
        if peaks_actor:
            peaks_actor.display(None, None, slices[2])
        view_up_vector = (0, 1, 0)

    # Generate the scene, set the camera and take the snapshot
    ren = window.Renderer()
    ren.add(volume_actor)
    if streamlines_actor:
        ren.add(streamlines_actor)
    elif peaks_actor:
        ren.add(peaks_actor)
    ren.set_camera(position=view_position,
                   view_up=view_up_vector,
                   focal_point=focal_point)

    window.snapshot(ren, size=(1920, 1080), offscreen=True,
                    fname=output_filename)


def create_marching_cube(data, color, opacity=0.5, min_threshold=0,
                         smoothing_iterations=None, spacing=[1., 1., 1.]):
    im = vtk.vtkImageData()
    I, J, K = data.shape[:3]
    im.SetDimensions(I, J, K)
    im.SetSpacing(spacing[0], spacing[1], spacing[2])
    im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    vol = np.swapaxes(data, 0, 2)
    vol = np.ascontiguousarray(vol)
    vol = vol.ravel()

    uchar_array = numpy_support.numpy_to_vtk(vol, deep=0)
    im.GetPointData().SetScalars(uchar_array)

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(im)
    mapper.Update()

    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(im)
    threshold.ThresholdByLower(min_threshold)
    threshold.ReplaceInOn()
    threshold.SetInValue(0)
    threshold.ReplaceOutOn()
    threshold.SetOutValue(1)
    threshold.Update()

    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(threshold.GetOutputPort())
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()

    mapper = vtk.vtkPolyDataMapper()

    if smoothing_iterations is not None:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(dmc.GetOutputPort())
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.Update()
        mapper.SetInputConnection(smoother.GetOutputPort())
    else:
        mapper.SetInputConnection(dmc.GetOutputPort())

    mapper.Update()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    mapper.ScalarVisibilityOff()

    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)

    return actor
