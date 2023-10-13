import numpy as np
from fury.utils import get_actor_from_polydata, numpy_to_vtk_image_data
import vtk


def create_tube_with_radii(positions, radii, error, error_coloring=False,
                           wireframe=False):
    # Generate the polydata from the centroids
    joint_count = len(positions)
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(joint_count)
    for j in range(joint_count):
        pts.InsertPoint(j, positions[j])
        lines.InsertCellPoint(j)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pts)
    polydata.SetLines(lines)

    # Generate the coloring from either the labels or the fitting error
    colors_arr = vtk.vtkFloatArray()
    for i in range(joint_count):
        if error_coloring:
            colors_arr.InsertNextValue(error[i])
        else:
            colors_arr.InsertNextValue(len(error) - 1 - i)
    colors_arr.SetName("colors")
    polydata.GetPointData().AddArray(colors_arr)

    # Generate the radii array for VTK
    radii_arr = vtk.vtkFloatArray()
    for i in range(joint_count):
        radii_arr.InsertNextValue(radii[i])
    radii_arr.SetName("radii")
    polydata.GetPointData().SetScalars(radii_arr)

    # Tube filter for the rendering with varying radii
    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputData(polydata)
    tubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tubeFilter.SetNumberOfSides(25)
    tubeFilter.CappingOn()

    # Map the coloring to the tube filter
    tubeFilter.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tubeFilter.GetOutputPort())
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("colors")
    if error_coloring:
        mapper.SetScalarRange(0, max(error))
    else:
        mapper.SetScalarRange(0, len(error))

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()

    return actor


def contour_actor_from_image(
    img, axis, slice_index,
    contour_value=1.,
    color=[255, 0, 0],
    opacity=1.,
    linewidth=3.,
    smoothing_radius=0.
):
    """
    Get an isocontour actor from an image slice, at a defined value.

    Parameters
    ----------
    img : Nifti1Image
        Nifti volume (mask, binary image, labels).
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    axis : int
        Slicing axis
    contour_values : float
        Values at which to extract isocontours.
    color : tuple, list of int
        Color of the contour in RGB [0, 255].
    opacity: float
        Opacity of the contour.
    linewidth : float
        Thickness of the contour line.
    smoothing_radius : float
        Pre-smoothing to apply to the image before 
        computing the contour (in pixels).

    Returns
    -------
    actor : vtkActor
        Actor for the contour polydata.
    """

    mask_data = numpy_to_vtk_image_data(
        np.rot90(img.get_fdata().take([slice_index], axis).squeeze()))
    mask_data.SetOrigin(0, 0, 0)

    if smoothing_radius > 0:
        smoother = vtk.vtkImageGaussianSmooth()
        smoother.SetRadiusFactor(smoothing_radius)
        smoother.SetDimensionality(2)
        smoother.SetInputData(mask_data)
        smoother.Update()
        mask_data = smoother.GetOutput()

    marching_squares = vtk.vtkMarchingSquares()
    marching_squares.SetInputData(mask_data)
    marching_squares.SetValue(0, contour_value)
    marching_squares.Update()

    actor = get_actor_from_polydata(marching_squares.GetOutput())
    actor.GetMapper().ScalarVisibilityOff()
    actor.GetProperty().SetLineWidth(linewidth)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)

    position =[0, 0, 0]
    position[axis] = slice_index

    if axis == 0:
        actor.SetOrientation(90, 0, 90)
    elif axis == 1:
        actor.SetOrientation(90, 0, 0)

    actor.SetPosition(*position)

    return actor