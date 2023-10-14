# -*- coding: utf-8 -*-


from fury import window

from scilpy.viz.backends.fury import create_scene
from scilpy.viz.backends.pil import (annotate_scene,
                                     create_canvas,
                                     draw_scene_at_pos,
                                     rgb2gray4pil)
from scilpy.viz.backends.vtk import contour_actor_from_image
from scilpy.viz.utils import compute_cell_topleft_pos
from scilpy.viz.slice import create_texture_slicer


def screenshot_volume(img, axis_name, slice_ids, size):
    """Take a screenshot of the given volume at the provided slice indices.

    Parameters
    ----------
    img : nib.Nifti1Image
        Volume image.
    axis_name : str
        Slicing axis name.
    slice_ids : array-like
        Slice indices.
    size : array-like
        Size of the screenshot image (pixels).

    Returns
    -------
    scene_container : list
        Scene screenshot data container.
    """

    scene_container = []

    for idx in slice_ids:

        slice_actor = create_texture_slicer(
            img.get_fdata(), axis_name, idx, offset=0.0
        )
        scene = create_scene([slice_actor], axis_name, idx, img.shape)
        scene_arr = window.snapshot(scene, size=size)
        scene_container.append(scene_arr)

    return scene_container


def screenshot_contour(bin_img, axis_name, slice_ids, size):
    """Take a screenshot of the given binary image countour with the 
    appropriate slice data at the provided slice indices.

    Parameters
    ----------
    bin_img : nib.Nifti1Image
        Binary volume image.
    axis_name : str
        Slicing axis name.
    slice_ids : array-like
        Slice indices.
    size : array-like
        Size of the screenshot image (pixels).

    Returns
    -------
    scene_container : list
        Scene screenshot data container.
    """
    scene_container = []

    if axis_name == "axial":
        ax_idx = 2
    elif axis_name == "coronal":
        ax_idx = 1
    elif axis_name == "sagittal":
        ax_idx = 0

    image_size_2d = list(bin_img.shape)
    image_size_2d[ax_idx] = 1

    for idx in slice_ids:
        actor = contour_actor_from_image(
            bin_img, ax_idx, idx, color=[255, 255, 255])

        scene = create_scene([actor], axis_name, idx, image_size_2d)
        scene_arr = window.snapshot(scene, size=size)
        scene_container.append(scene_arr)

    return scene_container


def compose_mosaic(
    img_scene_container,
    cell_size,
    rows,
    cols,
    slice_numbers,
    overlap_factor=None,
    transparency_scene_container=None,
    labelmap_scene_container=None,
    labelmap_overlay_alpha=0.7,
    mask_overlay_scene_container=None,
    mask_overlay_alpha=0.7,
    mask_overlay_color=None,
    vol_cmap_name=None,
    labelmap_cmap_name=None,
    display_slice_number=False,
    display_lr=False
):
    """Create the mosaic canvas for given number of rows and columns, and the
    requested cell size and overlap values.

    Parameters
    ----------
    img_scene_container : list
        Image scene data container.
    cell_size : array-like
        Cell size (pixels) (width, height).
    rows : int
        Row count.
    cols : int
        Column count.
    overlap_factor : array-like
        Overlap factor (horizontal, vertical).
    transparency_scene_container : list, optional
        Transaprency scene data container.
    labelmap_scene_container : list, optional
        Labelmap scene data container.
    mask_overlay_scene_container : list, optional
        Mask overlay scene data container.
    mask_overlay_alpha : float, optional
        Alpha value for mask overlay in range [0, 1].
    mask_overlay_color : list, optional
        Color for the mask overlay as a list of 3 integers in range [0, 255].
    vol_cmap_name : str, optional
        Colormap name for the image scene data.
    labelmap_cmap_name : str, optional
        Colormap name for the labelmap scene data.
    display_slice_number : bool, optional
        If true, displays the slice number in the upper left corner.
    display_lr : bool or int, optional
        If 1 or -1, identifies the left and right sides on the image. -1 flips 
        left and right positions.
    """

    def _compute_overlap_length(length, _overlap):
        return round(length * _overlap)

    cell_width = cell_size[0]
    cell_height = cell_size[1]

    overlap_h = overlap_v = 0
    if overlap_factor is not None:
        overlap_h = _compute_overlap_length(cell_width, overlap_factor[0])
        overlap_v = _compute_overlap_length(cell_width, overlap_factor[1])

    mosaic = create_canvas(*cell_size, overlap_h, overlap_v, rows, cols)

    offset_h = cell_width - overlap_h
    offset_v = cell_height - overlap_v
    from itertools import zip_longest
    for idx, (img_arr, trans_arr, labelmap_arr, mask_overlay_arr, slice_number) in enumerate(
            list(zip_longest(
                img_scene_container,
                transparency_scene_container,
                labelmap_scene_container,
                mask_overlay_scene_container,
                slice_numbers,
                fillvalue=tuple()))
    ):

        # Compute the mosaic cell position
        top_pos, left_pos = compute_cell_topleft_pos(
            idx, cols, offset_h, offset_v
        )

        # Convert the scene data to grayscale and adjust for handling with
        # Pillow
        _img_arr = rgb2gray4pil(img_arr)

        _trans_arr = None
        if len(trans_arr):
            _trans_arr = rgb2gray4pil(trans_arr)

        _labelmap_arr = None
        if len(labelmap_arr):
            _labelmap_arr = rgb2gray4pil(labelmap_arr)

        _mask_overlay_arr = None
        if len(mask_overlay_arr):
            _mask_overlay_arr = mask_overlay_arr

        # Draw the image (and labelmap overlay, if any) in the cell
        draw_scene_at_pos(
            mosaic,
            _img_arr,
            (cell_width, cell_height),
            left_pos,
            top_pos,
            transparency=_trans_arr,
            labelmap_overlay=_labelmap_arr,
            labelmap_overlay_alpha=labelmap_overlay_alpha,
            mask_overlay=_mask_overlay_arr,
            mask_overlay_alpha=mask_overlay_alpha,
            mask_overlay_color=mask_overlay_color,
            vol_cmap_name=vol_cmap_name,
            labelmap_cmap_name=labelmap_cmap_name,
        )

        annotate_scene(mosaic, slice_number, display_slice_number, display_lr)

    return mosaic
