# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
from fury import actor, window
from fury.utils import get_actor_from_polydata

from scilpy.utils.spatial import get_axis_index


class CamParams(Enum):
    """
    Enum containing camera parameters
    """

    VIEW_POS = 'view_position'
    VIEW_CENTER = 'view_center'
    VIEW_UP = 'up_vector'
    VIEW_ANGLE = 'view_angle'
    PARA_SCALE = 'parallel_scale'


def initialize_camera(orientation, slice_index, volume_shape, aspect_ratio):
    """
    Initialize a camera for a given orientation. The camera's focus
    (VIEW_CENTER) is set to the slice_index along the chosen orientation, at
    the center of the slice. The camera's position (VIEW_POS) is set
    perpendicular to the slice, at the origin along slice_index, pointing
    toward the slice's center.

    .. code-block:: text

         Camera                        Image plane
                         ---
        ---------         | VIEW_ANGLE      |
        |   *   |< -------|-----------------|* VIEW_CENTER
        ---------         |                 |
        VIEW_POS         ¯¯¯

    The camera's view angle (VIEW_ANGLE) is set to capture the smallest axis
    of the image plane in whole.

        - In perpective mode : 2 * arctan(ref_height / 2)
        - In parallel mode : ref_height / 2

    To compute ref_height, the slice's aspect ratio is compared to the
    viewport's. In case the slice's aspect ratio is greater, the reference
    height must be computed w.r.t the slice's width, scaled to the viewport's
    aspect ratio.

    Parameters
    ----------
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    volume_shape : tuple
        Shape of the sliced volume.
    aspect_ratio : float
        Ratio between viewport's width and height.

    Returns
    -------
    camera : dict
        Dictionnary containing camera information.
    """

    camera = {}
    axis_index = get_axis_index(orientation)
    volume_shape = volume_shape[:3]

    if slice_index is None:
        slice_index = volume_shape[axis_index] // 2

    # Set the camera's focus to the center of the slice
    camera[CamParams.VIEW_CENTER] = 0.5 * (np.array(volume_shape) - 1.0)
    camera[CamParams.VIEW_CENTER][axis_index] = slice_index

    # Set the camera's position perpendicular to the slice, at the origin
    # along slice_index, pointing toward the slice's center
    signed_view_pos = [-1.0, 1.0, -1.0]
    camera[CamParams.VIEW_POS] = 0.5 * (np.array(volume_shape) - 1.0)
    camera[CamParams.VIEW_POS][axis_index] = signed_view_pos[axis_index]

    # Set the camera's up vector parallel to the vertical axis
    # of the image w.r.t. the viewport
    vert_idx = 1 if axis_index == 2 else 2
    camera[CamParams.VIEW_UP] = np.zeros((3,))
    camera[CamParams.VIEW_UP][vert_idx] = -1.0

    # Based on : https://stackoverflow.com/questions/6565703/
    # math-algorithm-fit-image-to-screen-retain-aspect-ratio
    remain_axis = np.delete(volume_shape, [axis_index, vert_idx], 0)
    ref_height = volume_shape[vert_idx]
    if remain_axis[0] / volume_shape[vert_idx] > aspect_ratio:
        ref_height = remain_axis[0] / aspect_ratio

    # From vtkCamera documentation, see SetViewAngle and SetParallelScale
    # https://vtk.org/doc/nightly/html/classvtkCamera.html
    camera[CamParams.VIEW_ANGLE] = 2.0 * np.arctan(ref_height / 2.0)
    camera[CamParams.PARA_SCALE] = ref_height / (2.0)

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
    extents = np.vstack(([0, 0, 0], volume_shape[:3])).astype(int).T.flatten()

    if slice_index is None:
        slice_index = volume_shape[axis_index] // 2

    extents[2 * axis_index:2 * axis_index + 2] = slice_index
    slicer_actor.display_extent(*extents)


def set_viewport(scene, orientation, slice_index, volume_shape, aspect_ratio):
    """
    Place the camera in the scene to capture all its content at a given
    slice_index.

    Parameters
    ----------
    scene : window.Scene()
        Object from Fury containing the 3D scene.
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    volume_shape : tuple
        Shape of the sliced volume.
    aspect_ratio : float
        Ratio between viewport's width and height.
    """

    scene.projection('parallel')
    camera = initialize_camera(
        orientation, slice_index, volume_shape, aspect_ratio)
    scene.set_camera(position=camera[CamParams.VIEW_POS],
                     focal_point=camera[CamParams.VIEW_CENTER],
                     view_up=camera[CamParams.VIEW_UP])

    # View POS and View Angle do nothing for parallel projection.
    # To set the screen correctly, View POS is set to +-1 and the
    # parallel scale to half the largest planar axis (not orientation)
    scene.camera().SetParallelScale(camera[CamParams.PARA_SCALE])


def create_scene(actors, orientation, slice_index, volume_shape, aspect_ratio,
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
    aspect_ratio : float
        Ratio between viewport's width and height.
    bg_color: tuple
        Background color expressed as RGB triplet in the range [0, 1].

    Returns
    -------
    scene : window.Scene()
        Object from Fury containing the 3D scene.
    """

    scene = window.Scene()
    scene.background(bg_color)

    # Add actors to the scene
    for _actor in actors:
        scene.add(_actor)

    set_viewport(scene, orientation, slice_index, volume_shape, aspect_ratio)

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
    title : str
        Title of the scene. Defaults to Viewer.
    open_window : bool
        When true, initializes the interactor and opens the window
        (This suspends the current thread).

    Returns
    -------
    show_manager : window.ShowManager()
        Object from Fury containing the 3D scene interactor.
    """

    showm = window.ShowManager(scene, title=title,
                               size=window_size,
                               reset_camera=False,
                               interactor_style=interactor)
    showm.initialize()

    if open_window:
        showm.start()

    return showm


def snapshot_slices(actors, slice_ids, orientation, shape, size):
    """
    Snapshot a series of slice_ids from a scene on a given axis_name.

    Parameters
    ----------
    actors : list of vtkActor
        List of actors to snapshot.
    slice_ids : list of int
        List of slice indices to snapshot.
    orientation : str
        Name of the axis to snapshot.
    shape : tuple
        Shape of the volume.
    size : tuple
        Size of the viewport.

    Returns
    -------
    snapshots : generator of 2d np.ndarray
        Generator of snapshots.
    """

    scene = create_scene(actors, orientation, 0, shape, size[0] / size[1])

    for idx in slice_ids:
        for _actor in actors:
            set_display_extent(_actor, orientation, shape, idx)

        set_viewport(scene, orientation, idx, shape, size[0] / size[1])
        yield window.snapshot(scene, size=size).astype(np.uint8)


def snapshot_scenes(scenes, window_size):
    """
    Snapshot a list of scenes inside a window of given size.

    Parameters
    ----------
    scenes : list of window.Scene
        List of scenes to snapshot.
    window_size : tuple
        Size of the window.

    Returns
    -------
    snapshots : generator of 2d np.ndarray
        Generator of snapshots.
    """

    for scene in scenes:
        yield window.snapshot(scene, size=window_size)


def create_contours_actor(contours, opacity=1., linewidth=3.,
                          color=[255, 0, 0]):
    """
    Create an actor from a vtkPolyData of contours

    Parameters :
    ------------
    contours : vtkPolyData
        Contours polydata.
    opacity: float
        Opacity of the contour.
    linewidth : float
        Thickness of the contour line.
    color : tuple, list of int
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
    3-dimensional grid containing the SH coefficients of the fODF at each
    voxel, with the grid dimension having a size of 1 along the axis
    corresponding to the selected orientation.

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
    radial_scale : bool
        If True, enables radial scale for ODF slicer.
    norm : bool
        If True, enables normalization of ODF slicer.
    colormap : str, optional
        Colormap for the ODF slicer. If None, a RGB colormap is used.
    variance_k : float
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
            maximums = np.abs(np.append(sf_fodf, fodf_uncertainty,
                                        axis=-1)).max(axis=-1)

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


def create_peaks_actor(peaks, mask, opacity=1.0, linewidth=1.0, color=None,
                       symmetric=False, lut_values=None, lod=False,
                       lod_nb_points=10000, lod_points_size=3):
    """
    Create a Peaks actor from a N-dimensional array. Data can be from 2D
    (M 3D peaks) to 5D (XxYxZxM 3D peaks). Color is None by default so coloring
    defaults to orientation coloring.

    Parameters
    ----------
    peaks : np.ndarray
        Peaks data.
    mask : np.ndarray
        Mask used to restrict the rendered data.
    opacity : float
        Opacity of the peaks.
    linewidth : float
        Thickness of the peaks line.
    color : tuple, list of int
        Color of the peaks in RGB [0, 255]. If None, orientation
        coloring is used.
    symmetric : bool
        If True, the peaks are rendered symmetrically on both
        sides of the voxel's center.
    lut_values : np.ndarray, optional
        Use those values to color each peak.
    lod : bool
        If True, use level of detail rendering.
    lod_nb_points : int
        Number of points to use for level of detail rendering.
    lod_points_size : float
        Size of the points for level of detail rendering.

    Returns
    -------
    peaks_actor : actor.odf_slicer
        Fury object containing the peaks information.
    """

    return actor.peak_slicer(peaks, mask=mask, affine=np.eye(4),
                             colors=color, opacity=opacity,
                             linewidth=linewidth, symmetric=symmetric,
                             peaks_values=lut_values,
                             lod=lod, lod_points=lod_nb_points,
                             lod_points_size=lod_points_size)
