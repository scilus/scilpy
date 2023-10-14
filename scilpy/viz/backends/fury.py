# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
from fury import window

from scilpy.io.utils import snapshot
from scilpy.viz.backends.pil import (create_canvas,
                                     draw_scene_at_pos,
                                     rgb2gray4pil)


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
        # Tighten the view around the data
        camera[CamParams.ZOOM_FACTOR] = 2.0 / max(volume_shape[1:])
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
        # Tighten the view around the data
        camera[CamParams.ZOOM_FACTOR] = 2.0 / max(
            [volume_shape[0], volume_shape[2]])
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
        # Tighten the view around the data
        camera[CamParams.ZOOM_FACTOR] = 2.0 / max(volume_shape[:2])
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


def create_scene(actors, orientation, slice_index,
                 volume_shape, bg_color=(0, 0, 0)):
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
    for curr_actor in actors:
        scene.add(curr_actor)

    return scene


def create_interactive_window(scene, window_size, interactor, title, 
                              open_window=True):
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


def render_scene(scene, window_size, interactor,
                 output, silent, mask_scene=None, title='Viewer'):
    """
    Render a scene. If a output is supplied, a snapshot of the rendered
    scene is taken. If a mask is supplied, all values outside the mask are set
    to full transparency in the saved scene.

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
    mask_scene : window.Scene(), optional
        Transparency mask scene.
    title : str, optional
        Title of the scene. Defaults to Viewer.
    """
    if not silent:
        create_interactive_window(scene, window_size, interactor, title)

    if output:
        if mask_scene is not None:
            # Create the screenshots
            scene_arr = window.snapshot(scene, size=window_size)
            mask_scene_arr = window.snapshot(mask_scene, size=window_size)
            # Create the target image
            out_img = create_canvas(*window_size, 0, 0, 1, 1)
            # Convert the mask scene data to grayscale and adjust for handling
            # with Pillow
            _mask_arr = rgb2gray4pil(mask_scene_arr)
            # Create the masked image
            draw_scene_at_pos(
                out_img, scene_arr, window_size, 0, 0, mask=_mask_arr
            )

            out_img.save(output)
        else:
            snapshot(scene, output, size=window_size)
