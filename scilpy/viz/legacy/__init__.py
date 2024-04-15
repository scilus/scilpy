

def display_slices(volume_actor, slices,
                   output_filename, axis_name,
                   view_position, focal_point,
                   peaks_actor=None, streamlines_actor=None,
                   roi_actors=None):
    """
    Display the slices of the volume actor and take a snapshot. This
    function is a legacy function and should not be used in new code.
    Instead, use the functions in :py:mod:`~scilpy.viz.screenshot` for
    screenshotting, or in :py:mod:`~scilpy.viz.slice` for volume slicing
    using vtk and fury.

    Parameters
    ----------
    volume_actor : Volume
        The volume actor to display.
    slices : tuple of int
        The slices to display.
    output_filename : str
        The output filename.
    axis_name : str
        The axis name. See :py:const:`~scilpy.utils.spatial.RAS_AXES_NAMES`.
    view_position : tuple of float
        The view position.
    focal_point : tuple of float
        The focal point.
    peaks_actor : vtkActor, optional
        The peaks actor to display.
    streamlines_actor : vtkActor, optional
        The streamlines actor to display.
    roi_actors : list of vtkActor, optional
        The ROI actors to display.
    """
    from scilpy.viz.backends.fury import snapshot_scenes
    from scilpy.viz.screenshot import compose_image
    from fury import window

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

    # Legacy. When this snapshotting gets updated to align with the
    # viz module, snapshot_scenes should be called directly
    snapshot = next(snapshot_scenes([scene], (1920, 1080)))
    img = compose_image(snapshot, (1920, 1080), "NONE")
    img.save(output_filename)
