# -*- coding: utf-8 -*-

from fury import window

from scilpy.io.utils import snapshot


def display_slices(volume_actor, slices,
                   output_filename, axis_name,
                   view_position, focal_point,
                   peaks_actor=None, streamlines_actor=None,
                   roi_actors=None):
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
    scene = window.Scene()
    scene.add(volume_actor)
    if streamlines_actor:
        scene.add(streamlines_actor)
    elif peaks_actor:
        scene.add(peaks_actor)

    if roi_actors:
        for roi_actor in roi_actors:
            scene.add(roi_actor)
    scene.set_camera(position=view_position,
                     view_up=view_up_vector,
                     focal_point=focal_point)

    snapshot(scene, output_filename, size=(1920, 1080), offscreen=True)
