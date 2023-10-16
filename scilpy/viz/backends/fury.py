# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
from fury import actor, window
from fury.utils import get_actor_from_polydata

from scilpy.utils.util import get_axis_index


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
    # heuristic for setting the camera position at a distance
    # proportional to the scale of the scene
    eye_distance = max(volume_shape)
    axis_index = get_axis_index(orientation)

    if slice_index is None:
        slice_index = volume_shape[axis_index] // 2

    view_pos_sign = [-1.0, 1.0, -1.0]
    camera[CamParams.VIEW_POS] = 0.5 * (np.array(volume_shape) - 1.0)
    camera[CamParams.VIEW_POS][axis_index] = (
        view_pos_sign[axis_index] * eye_distance)

    camera[CamParams.VIEW_CENTER] = 0.5 * (np.array(volume_shape) - 1.0)
    camera[CamParams.VIEW_CENTER][axis_index] = slice_index

    camera[CamParams.VIEW_UP] = np.array([0.0, 0.0, 1.0])
    if axis_index == 2:
        camera[CamParams.VIEW_UP] = np.array([0.0, 1.0, 0.0])

    camera[CamParams.ZOOM_FACTOR] = 2.0 / \
        min(np.delete(volume_shape, axis_index, 0))

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

    axis_index = get_axis_index(orientation)
    extents = np.vstack(([0., 0., 0.], volume_shape)).T.flatten()

    if slice_index is None:
        slice_index = volume_shape[axis_index] // 2

    extents[2 * axis_index:2 * axis_index + 2] = slice_index
    slicer_actor.display_extent(*extents)


def create_scene(actors, orientation, slice_index, volume_shape,
                 bg_color=(0, 0, 0)):
    """
    Create a 3D scene containing actors fitting inside a grid. The camera is
    placed based on the orientation supplied by the user. The projection mode
    is parallel.

    Parameters
    ----------
    actors : list of actor
        Ensemble of actors from Fury
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    volume_shape : tuple
        Shape of the sliced volume.
    bg_color: tuple, optional
        Background color expressed as RGB triplet in the range [0, 1].

    Returns
    -------
    scene : window.Scene()
        Object from Fury containing the 3D scene.
    """
    # Configure camera
    camera = initialize_camera(orientation, slice_index, volume_shape)

    scene = window.Scene()
    scene.background(bg_color)
    scene.projection('parallel')
    scene.set_camera(position=camera[CamParams.VIEW_POS],
                     focal_point=camera[CamParams.VIEW_CENTER],
                     view_up=camera[CamParams.VIEW_UP])
    scene.zoom(camera[CamParams.ZOOM_FACTOR])

    # Add actors to the scene
    for _actor in actors:
        scene.add(_actor)

    return scene


def create_interactive_window(scene, window_size, interactor,
                              title="Viewer", open_window=True):
    """
    Create a 3D window with the content of scene, equiped with an interactor.

    Parameters
    ----------
    scene : window.Scene()
        Object from Fury containing the 3D scene.
    window_size : tuple (width, height)
        The dimensions for the vtk window.
    interactor : str
        Specify interactor mode for vtk window. Choices are image or trackball.
    title : str, optional
        Title of the scene. Defaults to Viewer.
    open_window : bool, optional
        When true, initializes the interactor and opens the window 
        (This suspends the current thread).
    """
    showm = window.ShowManager(scene, title=title,
                               size=window_size,
                               reset_camera=False,
                               interactor_style=interactor)

    if open_window:
        showm.initialize()
        showm.start()

    return showm


def snapshot_scenes(scenes, window_size):
    """
    Snapshot a list of scenes inside a window of given size
    """
    return [window.snapshot(scene, size=window_size) for scene in scenes]


def create_contours_actor(contours, opacity=1., linewidth=3.,
                          color=[255, 0, 0]):
    """
    Create an actor from a vtkPolyData of contours

    Parameters :
    ------------
    contours : vtkPolyData
        Contours polydata.
    opacity: float, optional
        Opacity of the contour.
    linewidth : float, optional
        Thickness of the contour line.
    color : tuple, list of int, optional
        Color of the contour in RGB [0, 255].

    Returns
    -------
    contours_actor : actor.odf_slicer
        Fury object containing the contours information.
    """

    contours_actor = get_actor_from_polydata(contours)
    contours_actor.GetMapper().ScalarVisibilityOff()
    contours_actor.GetProperty().SetLineWidth(linewidth)
    contours_actor.GetProperty().SetColor(color)
    contours_actor.GetProperty().SetOpacity(opacity)

    return contours_actor


def create_odf_actors(sf_fodf, sphere, scale, sf_variance=None, mask=None,
                      radial_scale=False, norm=False, colormap=None,
                      variance_k=1.0, variance_color=None):
    """
    Create a ODF slicer actor displaying a fODF slice. The input volume is a
    3-dimensional grid containing the SH coefficients of the fODF for each
    voxel at each voxel, with the grid dimension having a size of 1 along the
    axis corresponding to the selected orientation.

    Parameters
    ----------
    sf_fodf : np.ndarray
        Spherical function of fODF data.
    sphere: DIPY Sphere
        Sphere used for visualization.
    scale : float
        Scaling factor for FODF.
    sf_variance : np.ndarray, optional
        Spherical function of the variance fODF data.
    mask : np.ndarray, optional
        Only the data inside the mask will be displayed. Defaults to None.
    radial_scale : bool, optional
        If True, enables radial scale for ODF slicer.
    norm : bool, optional
        If True, enables normalization of ODF slicer.
    colormap : str, optional
        Colormap for the ODF slicer. If None, a RGB colormap is used.
    variance_k : float, optional
        Factor that multiplies sqrt(variance).
    variance_color : tuple, optional
        Color of the variance fODF data, in RGB.

    Returns
    -------
    odf_actor : actor.odf_slicer
        Fury object containing the odf information.
    var_actor : actor.odf_slicer
        Fury object containing the odf variance information.
    """

    var_actor = None
    if sf_variance is not None:
        fodf_uncertainty = sf_fodf + variance_k * np.sqrt(
            np.clip(sf_variance, 0, None))

        # normalise fodf and variance
        if norm:
            maximums = np.abs(np.append(sf_fodf, fodf_uncertainty, axis=-1)) \
                .max(axis=-1)
            sf_fodf[maximums > 0] /= maximums[maximums > 0][..., None]
            fodf_uncertainty[maximums > 0] /= maximums[maximums > 0][..., None]

        var_actor = actor.odf_slicer(fodf_uncertainty, mask=mask, norm=False,
                                     radial_scale=radial_scale,
                                     sphere=sphere, scale=scale,
                                     colormap=variance_color)

        var_actor.GetProperty().SetDiffuse(0.0)
        var_actor.GetProperty().SetAmbient(1.0)
        var_actor.GetProperty().SetFrontfaceCulling(True)

    odf_actor = actor.odf_slicer(sf_fodf, mask=mask, norm=False,
                                 radial_scale=radial_scale,
                                 sphere=sphere, scale=scale,
                                 colormap=colormap)

    return odf_actor, var_actor
