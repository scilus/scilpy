# -*- coding: utf-8 -*-

from fury import window


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
