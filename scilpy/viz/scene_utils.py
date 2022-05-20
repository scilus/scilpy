# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np

from dipy.reconst.shm import sh_to_sf_matrix
from fury import window, actor
from fury.colormap import distinguishable_colormap
import vtk

from scilpy.io.utils import snapshot
from scilpy.reconst.bingham import bingham_to_sf


class CamParams(Enum):
    """
    Enum containing camera parameters
    """
    VIEW_POS = 'view_position'
    VIEW_CENTER = 'view_center'
    VIEW_UP = 'up_vector'
    ZOOM_FACTOR = 'zoom_factor'


def initialize_camera(orientation, slice_index, volume_shape):
    """
    Initialize a camera for a given orientation.

    Parameters
    ----------
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    volume_shape : tuple
        Shape of the sliced volume.

    Returns
    -------
    camera : dict
        Dictionnary containing camera information.
    """
    camera = {}
    # Tighten the view around the data
    camera[CamParams.ZOOM_FACTOR] = 2.0 / max(volume_shape)
    # heuristic for setting the camera position at a distance
    # proportional to the scale of the scene
    eye_distance = max(volume_shape)
    if orientation == 'sagittal':
        if slice_index is None:
            slice_index = volume_shape[0] // 2
        camera[CamParams.VIEW_POS] = np.array([-eye_distance,
                                               (volume_shape[1] - 1) / 2.0,
                                               (volume_shape[2] - 1) / 2.0])
        camera[CamParams.VIEW_CENTER] = np.array([slice_index,
                                                  (volume_shape[1] - 1) / 2.0,
                                                  (volume_shape[2] - 1) / 2.0])
        camera[CamParams.VIEW_UP] = np.array([0.0, 0.0, 1.0])
    elif orientation == 'coronal':
        if slice_index is None:
            slice_index = volume_shape[1] // 2
        camera[CamParams.VIEW_POS] = np.array([(volume_shape[0] - 1) / 2.0,
                                               eye_distance,
                                               (volume_shape[2] - 1) / 2.0])
        camera[CamParams.VIEW_CENTER] = np.array([(volume_shape[0] - 1) / 2.0,
                                                  slice_index,
                                                  (volume_shape[2] - 1) / 2.0])
        camera[CamParams.VIEW_UP] = np.array([0.0, 0.0, 1.0])
    elif orientation == 'axial':
        if slice_index is None:
            slice_index = volume_shape[2] // 2
        camera[CamParams.VIEW_POS] = np.array([(volume_shape[0] - 1) / 2.0,
                                               (volume_shape[1] - 1) / 2.0,
                                               -eye_distance])
        camera[CamParams.VIEW_CENTER] = np.array([(volume_shape[0] - 1) / 2.0,
                                                  (volume_shape[1] - 1) / 2.0,
                                                  slice_index])
        camera[CamParams.VIEW_UP] = np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError('Invalid axis name: {0}'.format(orientation))
    return camera


def set_display_extent(slicer_actor, orientation, volume_shape, slice_index):
    """
    Set the display extent for a fury actor in ``orientation``.

    Parameters
    ----------
    slicer_actor : actor
        Slicer actor from Fury
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    volume_shape : tuple
        Shape of the sliced volume.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    """
    if orientation == 'sagittal':
        if slice_index is None:
            slice_index = volume_shape[0] // 2
        slicer_actor.display_extent(slice_index, slice_index,
                                    0, volume_shape[1],
                                    0, volume_shape[2])
    elif orientation == 'coronal':
        if slice_index is None:
            slice_index = volume_shape[1] // 2
        slicer_actor.display_extent(0, volume_shape[0],
                                    slice_index, slice_index,
                                    0, volume_shape[2])
    elif orientation == 'axial':
        if slice_index is None:
            slice_index = volume_shape[2] // 2
        slicer_actor.display_extent(0, volume_shape[0],
                                    0, volume_shape[1],
                                    slice_index, slice_index)
    else:
        raise ValueError('Invalid axis name : {0}'.format(orientation))


def create_odf_slicer(sh_fodf, orientation, slice_index, mask, sphere,
                      nb_subdivide, sh_order, sh_basis, full_basis,
                      scale, radial_scale, norm, colormap):
    """
    Create a ODF slicer actor displaying a fODF slice. The input volume is a
    3-dimensional grid containing the SH coefficients of the fODF for each
    voxel at each voxel, with the grid dimension having a size of 1 along the
    axis corresponding to the selected orientation.

    Parameters
    ----------
    sh_fodf : np.ndarray
        Spherical harmonics of fODF data.
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    mask : np.ndarray, optional
        Only the data inside the mask will be displayed. Defaults to None.
    sphere: DIPY Sphere
        Sphere used for visualization.
    nb_subdivide : int
        Number of subdivisions for given sphere. If None, uses the given sphere
        as is.
    sh_order : int
        Maximum spherical harmonics order.
    sh_basis : str
        Type of basis for the spherical harmonics.
    full_basis : bool
        Boolean indicating if the basis is full or not.
    scale : float
        Scaling factor for FODF.
    radial_scale : bool
        If True, enables radial scale for ODF slicer.
    norm : bool
        If True, enables normalization of ODF slicer.
    colormap : str
        Colormap for the ODF slicer. If None, a RGB colormap is used.

    Returns
    -------
    odf_actor : actor.odf_slicer
        Fury object containing the odf information.
    """
    # Subdivide the spheres if nb_subdivide is provided
    if nb_subdivide is not None:
        sphere = sphere.subdivide(nb_subdivide)

    # SH coefficients to SF coefficients matrix
    B_mat = sh_to_sf_matrix(sphere, sh_order, sh_basis,
                            full_basis, return_inv=False)

    odf_actor = actor.odf_slicer(sh_fodf, mask=mask, norm=norm,
                                 radial_scale=radial_scale,
                                 sphere=sphere,
                                 colormap=colormap,
                                 scale=scale, B_matrix=B_mat)
    set_display_extent(odf_actor, orientation, sh_fodf.shape[:3], slice_index)

    return odf_actor


def _get_affine_for_texture(orientation, offset):
    """
    Get the affine transformation to apply to the texture
    to offset it from the fODF grid.

    Parameters
    ----------
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    offset : float
        The offset of the texture image.

    Returns
    -------
    affine : np.ndarray
        The affine transformation.
    """
    if orientation == 'sagittal':
        v = np.array([offset, 0.0, 0.0])
    elif orientation == 'coronal':
        v = np.array([0.0, -offset, 0.0])
    elif orientation == 'axial':
        v = np.array([0.0, 0.0, offset])
    else:
        raise ValueError('Invalid axis name : {0}'.format(orientation))

    affine = np.identity(4)
    affine[0:3, 3] = v
    return affine


def create_texture_slicer(texture, orientation, slice_index, mask=None,
                          value_range=None, opacity=1.0, offset=0.5,
                          interpolation='nearest'):
    """
    Create a texture displayed behind the fODF. The texture is applied on a
    plane with a given offset for the fODF grid.

    Parameters
    ----------
    texture : np.ndarray (3d or 4d)
        Texture image. Can be 3d for scalar data of 4d for RGB data, in which
        case the values must be between 0 and 255.
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    mask : np.ndarray, optional
        Only the data inside the mask will be displayed. Defaults to None.
    value_range : tuple (2,), optional
        The range of values mapped to range [0, 1] for the texture image. If
        None, it equals to (bg.min(), bg.max()). Defaults to None.
    opacity : float, optional
        The opacity of the texture image. Opacity of 0.0 means transparent and
        1.0 is completely visible. Defaults to 1.0.
    offset : float, optional
        The offset of the texture image. Defaults to 0.5.
    interpolation : str, optional
        Interpolation mode for the texture image. Choices are nearest or
        linear. Defaults to nearest.

    Returns
    -------
    slicer_actor : actor.slicer
        Fury object containing the texture information.
    """
    affine = _get_affine_for_texture(orientation, offset)

    if mask is not None:
        texture[np.where(mask == 0)] = 0

    if value_range:
        texture = np.clip((texture - value_range[0]) / value_range[1] * 255,
                          0, 255)

    slicer_actor = actor.slicer(texture, affine=affine,
                                opacity=opacity, interpolation=interpolation)
    set_display_extent(slicer_actor, orientation, texture.shape, slice_index)
    return slicer_actor


def create_peaks_slicer(data, orientation, slice_index, peak_values=None,
                        mask=None, color=None, peaks_width=1.0,
                        symmetric=False):
    """
    Create a peaks slicer actor rendering a slice of the fODF peaks

    Parameters
    ----------
    data : np.ndarray
        Peaks data.
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    peak_values : np.ndarray, optional
        Peaks values. Defaults to None.
    mask : np.ndarray, optional
        Only the data inside the mask will be displayed. Defaults to None.
    color : tuple (3,), optional
        Color used for peaks. If None, a RGB colormap is used. Defaults to
        None.
    peaks_width : float, optional
        Width of peaks segments. Defaults to 1.0.
    symmetric : bool, optional
        If True, peaks are drawn for both peaks_dirs and -peaks_dirs. Else,
        peaks are only drawn for directions given by peaks_dirs. Defaults to
        False.

    Returns
    -------
    slicer_actor : actor.peak_slicer
        Fury object containing the peaks information.
    """
    # Normalize input data
    norm = np.linalg.norm(data, axis=-1)
    data[norm > 0] /= norm[norm > 0].reshape((-1, 1))

    # Instantiate peaks slicer
    peaks_slicer = actor.peak_slicer(data, peaks_values=peak_values,
                                     mask=mask, colors=color,
                                     linewidth=peaks_width,
                                     symmetric=symmetric)
    set_display_extent(peaks_slicer, orientation, data.shape, slice_index)

    return peaks_slicer


def create_bingham_slicer(data, orientation, slice_index,
                          sphere, color_per_lobe=False):
    """
    Create a bingham fit slicer using a combination of odf_slicer actors

    Parameters
    ----------
    data: ndarray (X, Y, Z, 9 * nb_lobes)
        The Bingham volume.
    orientation: str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index: int
        Index of the slice of interest along the chosen orientation.
    sphere: DIPY Sphere
        Sphere used for visualization.
    color_per_lobe: bool, optional
        If true, each Bingham distribution is colored using a disting color.
        Else, Bingham distributions are colored by their orientation.

    Return
    ------
    actors: list of fury odf_slicer actors
        ODF slicer actors representing the Bingham distributions.
    """
    shape = data.shape
    nb_lobes = shape[-2]
    colors = [c * 255 for i, c in zip(range(nb_lobes),
                                      distinguishable_colormap())]

    # lmax norm for normalization
    lmaxnorm = np.max(np.abs(data[..., 0]), axis=-1)
    bingham_sf = bingham_to_sf(data, sphere.vertices)

    actors = []
    for nn in range(nb_lobes):
        sf = bingham_sf[..., nn, :]
        sf[lmaxnorm > 0] /= lmaxnorm[lmaxnorm > 0][:, None]
        color = colors[nn] if color_per_lobe else None
        odf_actor = actor.odf_slicer(sf, sphere=sphere, norm=False,
                                     colormap=color)
        set_display_extent(odf_actor, orientation, shape[:3], slice_index)
        actors.append(odf_actor)

    return actors


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


def create_scene(actors, orientation, slice_index, volume_shape):
    """
    Create a 3D scene containing actors fitting inside a grid. The camera is
    placed based on the orientation supplied by the user. The projection mode
    is parallel.

    Parameters
    ----------
    actors : tab
        Ensemble of actors from Fury
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    volume_shape : tuple
        Shape of the sliced volume.

    Returns
    -------
    scene : window.Scene()
        Object from Fury containing the 3D scene.
    """
    # Configure camera
    camera = initialize_camera(orientation, slice_index, volume_shape)

    scene = window.Scene()
    scene.projection('parallel')
    scene.set_camera(position=camera[CamParams.VIEW_POS],
                     focal_point=camera[CamParams.VIEW_CENTER],
                     view_up=camera[CamParams.VIEW_UP])
    scene.zoom(camera[CamParams.ZOOM_FACTOR])

    # Add actors to the scene
    for curr_actor in actors:
        scene.add(curr_actor)

    return scene


def render_scene(scene, window_size, interactor,
                 output, silent, title='Viewer'):
    """
    Render a scene. If a output is supplied, a snapshot of the rendered
    scene is taken.

    Parameters
    ----------
    scene : window.Scene()
        3D scene to render.
    window_size : tuple (width, height)
        The dimensions for the vtk window.
    interactor : str
        Specify interactor mode for vtk window. Choices are image or trackball.
    output : str
        Path to output file.
    silent : bool
        If True, disable interactive visualization.
    title : str, optional
        Title of the scene. Defaults to Viewer.
    """
    if not silent:
        showm = window.ShowManager(scene, title=title,
                                   size=window_size,
                                   reset_camera=False,
                                   interactor_style=interactor)

        showm.initialize()
        showm.start()

    if output:
        snapshot(scene, output, size=window_size)
