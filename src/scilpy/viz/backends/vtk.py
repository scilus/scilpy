# -*- coding: utf-8 -*-

from fury.utils import numpy_to_vtk_image_data
import vtk


def get_color_by_name(color_name):
    """
    Get a vtkColor by name. See :
        https://vtk.org/doc/nightly/html/classvtkNamedColors.html

    Some color names can be found in the CSS3 specification :
        https://www.w3.org/TR/css-color-3/#html4
        https://www.w3.org/TR/css-color-3/#svg-color

    Parameters
    ----------
    color_name : str
        Name of the color.

    Returns
    -------
    color : vtkColor
        RGB color object.
    """

    try:
        color_wheel = vtk.vtkNamedColors()
        return color_wheel.GetColor3d(color_name)
    except Exception:
        raise ValueError("Invalid VTK color name : {}".format(color_name))


def lut_from_colors(colors, value_range):
    """
    Create a linear VTK lut from a list of colors and a range of values.

    Parameters
    ----------
    colors : list
        List of colors (grayscale or RGB).
    value_range : tuple
        Range of values to map the colors to.

    Returns
    -------
    lut : vtkLookupTable
        VTK lookup table.
    """
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(len(colors))
    lut.SetTableRange(*value_range)
    lut.Build()

    _cl = vtk.vtkUnsignedCharArray()
    _cl.SetNumberOfComponents(len(colors[0]))
    _cl.SetNumberOfTuples(len(colors))
    for i, _v in enumerate(colors):
        _cl.SetTuple(i, _v)

    lut.SetTable(_cl)

    return lut


def create_tube_with_radii(positions, radii, error, error_coloring=False,
                           wireframe=False):
    """
    Create a tube actor from a list of positions, radii and errors.

    Parameters
    ----------
    positions : np.ndarray
        Array of positions of the joints.
    radii : np.ndarray
        Array of radii at the joints.
    error : np.ndarray
        Array of fitting error at the joints.
    error_coloring : bool, optional
        Color the tube based on the amplitude of the error.
    wireframe : bool, optional
        Render the tube as a wireframe.

    Returns
    -------
    actor : vtkActor
        Tube actor.
    """

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


def contours_from_data(data, contour_values=[1.], smoothing_radius=0.):
    """
    Get isocontour polydata from an array, at defined values.

    Parameters
    ----------
    data : np.ndarray
        N-dimensional array of data (mask, binary image, labels).
    contour_values : list
        Values at which to extract isocontours.
    smoothing_radius : float
        Pre-smoothing to apply to the image before
        computing the contour (in pixels).

    Returns
    -------
    contours : vtkPolyData
       Contours polydata.
    """

    mask_data = numpy_to_vtk_image_data(data)
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

    marching_squares.SetNumberOfContours(len(contour_values))
    for i, value in enumerate(contour_values):
        marching_squares.SetValue(i, value)
    marching_squares.Update()

    return marching_squares.GetOutput()
