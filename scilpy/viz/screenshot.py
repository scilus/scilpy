# -*- coding: utf-8 -*-

from fury import window
from scilpy.utils.util import get_axis_index

from scilpy.viz.backends.fury import (create_peaks_actor, create_scene,
                                      set_display_extent,
                                      set_viewport,
                                      snapshot_slices)
from scilpy.viz.backends.pil import (annotate_image,
                                     create_canvas,
                                     draw_2d_array_at_position,
                                     any2grayscale)
from scilpy.viz.color import get_lookup_table
from scilpy.viz.utils import compute_cell_topleft_pos
from scilpy.viz.slice import create_peaks_slicer, create_texture_slicer, create_contours_slicer


def screenshot_volume(img, orientation, slice_ids, size):
    """
    Take a screenshot of the given volume at the provided slice indices.

    Parameters
    ----------
    img : nib.Nifti1Image
        Volume image.
    orientation : str
        Slicing axis name.
    slice_ids : array-like
        Slice indices.
    size : array-like
        Size of the screenshot image (pixels).

    Returns
    -------
    snapshots : generator
        Scene screenshots generator.
    """

    slice_actor = create_texture_slicer(img.get_fdata(), orientation, 0,
                                        offset=0.0)

    return snapshot_slices([slice_actor], slice_ids, orientation,
                           img.shape, size)


def screenshot_contour(bin_img, orientation, slice_ids, size, bg_opacity=0.):
    """
    Take a screenshot of the given binary image countour with the
    appropriate slice data at the provided slice indices.

    Parameters
    ----------
    bin_img : nib.Nifti1Image
        Binary volume image.
    orientation : str
        Slicing axis name.
    slice_ids : array-like
        Slice indices.
    size : array-like
        Size of the screenshot image (pixels).
    bg_opacity : float, optional
        Background opacity in range [0, 1].

    Returns
    -------
    snapshots : generator
        Scene screenshots generator.
    """

    ax_idx = get_axis_index(orientation)
    image_size_2d = list(bin_img.shape)
    image_size_2d[ax_idx] = 1

    _actors = []
    if bg_opacity > 0.:
        _actors.append(create_texture_slicer(bin_img.get_fdata(),
                                             orientation, 0,
                                             offset=0.0, opacity=bg_opacity))

    scene = create_scene(_actors, orientation, 0, image_size_2d,
                         size[0] / size[1])

    for idx in slice_ids:
        for _actor in _actors:
            set_display_extent(_actor, orientation, image_size_2d, idx)

        contour_actor = create_contours_slicer(
            bin_img.get_fdata(), [1.], ax_idx, idx, color=[255, 255, 255])

        scene.add(contour_actor)
        set_viewport(scene, orientation, idx, image_size_2d, size[0] / size[1])

        yield window.snapshot(scene, size=size)
        scene.rm(contour_actor)


def screenshot_peaks(img, orientation, slice_ids, size, mask_img=None):
    """
    Take a screenshot of the given peaks image at the provided slice indices.

    Parameters
    ----------
    img : nib.Nifti1Image
        Peaks volume image.
    orientation : str
        Slicing axis name.
    slice_ids : array-like
        Slice indices.
    size : array-like
        Size of the screenshot image (pixels).

    Returns
    -------
    snapshots : generator
        Scene screenshots generator.
    """

    peaks_actor = create_peaks_slicer(img.get_fdata(), orientation, 0,
                                      mask=mask_img.get_fdata().astype(bool))

    return snapshot_slices([peaks_actor], slice_ids, orientation,
                           img.shape, size)


def compose_image(img_scene, img_size, slice_number, corner_position=(0, 0),
                  transparency_scene=None, labelmap_scene=None,
                  labelmap_overlay_alpha=0.7, mask_overlay_scene=None,
                  mask_overlay_alpha=0.7, mask_overlay_color=None,
                  peaks_overlay_scene=None, peaks_overlay_alpha=0.7,
                  vol_cmap_name=None, labelmap_cmap_name=None,
                  display_slice_number=False, display_lr=False,
                  lr_labels=["L", "R"], canvas=None):
    """
    Compose an image with the given scenes, with transparency, overlays,
    labelmap and annotations. If no canvas for the image is given, it will
    be automatically create with sizings to fit.

    Parameters
    ----------
    img_scene : np.ndarray
        Image scene data.
    img_size : array-like
        Image size (pixels) (width, height).
    slice_number : int
        Number of the current slice.
    corner_position : array-like, optional
        Image corner (pixels) (left, top).
    transparency_scene : np.ndarray, optional
        Transaprency scene data.
    labelmap_scene : np.ndarray, optional
        Labelmap scene data.
    labelmap_alpha : float, optional
        Alpha value for labelmap overlay in range [0, 1].
    mask_overlay_scene : np.ndarray, optional
        Mask overlay scene data.
    mask_overlay_alpha : float, optional
        Alpha value for mask overlay in range [0, 1].
    mask_overlay_color : list, optional
        Color for the mask overlay as a list of 3 integers in range [0, 255].
    peaks_overlay_scene : np.ndarray, optional
        Peaks overlay scene data.
    peaks_overlay_alpha : float, optional
        Alpha value for peaks overlay in range [0, 1].
    vol_cmap_name : str, optional
        Colormap name for the image scene data.
    labelmap_cmap_name : str, optional
        Colormap name for the labelmap scene data.
    display_slice_number : bool, optional
        If true, displays the slice number in the upper left corner.
    display_lr : bool or int, optional
        If 1 or -1, annotates the left and right sides on the image. -1 flips
        left and right positions.
    lr_labels : list, optional
        Labels used to annotate the left and right sides of the image.
    canvas : PIL.Image, optional
        Base canvas into which to paste the scene.

    Returns
    -------
    canvas : PIL.Image
        Canvas containing the pasted scene.
    """

    if canvas is None:
        canvas = create_canvas(*img_size, 1, 1, 0, 0)

    vol_lut, labelmap_lut = None, None
    if vol_cmap_name:
        vol_lut = get_lookup_table(vol_cmap_name)
    if labelmap_cmap_name:
        labelmap_lut = get_lookup_table(labelmap_cmap_name)

    draw_2d_array_at_position(canvas, img_scene, img_size,
                              corner_position[0], corner_position[1],
                              transparency=transparency_scene,
                              labelmap_overlay=labelmap_scene,
                              labelmap_overlay_alpha=labelmap_overlay_alpha,
                              mask_overlay=mask_overlay_scene,
                              mask_overlay_alpha=mask_overlay_alpha,
                              mask_overlay_color=mask_overlay_color,
                              peaks_overlay=peaks_overlay_scene,
                              peaks_overlay_alpha=peaks_overlay_alpha,
                              vol_lut=vol_lut, labelmap_lut=labelmap_lut)

    annotate_image(canvas, slice_number, display_slice_number,
                   display_lr, lr_labels)

    return canvas


def compose_mosaic(img_scene_container, cell_size, rows, cols, slice_numbers,
                   overlap_factor=None, transparency_scene_container=None,
                   labelmap_scene_container=None, labelmap_overlay_alpha=0.7,
                   mask_overlay_scene_container=None, mask_overlay_alpha=0.7,
                   mask_overlay_color=None, vol_cmap_name=None,
                   labelmap_cmap_name=None, display_slice_number=False,
                   display_lr=False, lr_labels=["L", "R"]):
    """
    Create the mosaic canvas for given number of rows and columns,
    and the requested cell size and overlap values.

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
        If 1 or -1, annotates the left and right sides on the image. -1 flips
        left and right positions.
    lr_labels : list, optional
        Labels used to annotate the left and right sides of the image.

    Returns
    -------
    mosaic : PIL.Image
        Canvas containing the mosaic scene.
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
    for idx, (img_arr, trans_arr, labelmap_arr,
              mask_overlay_arr, slice_number) in enumerate(
                                            list(zip_longest(
                                                img_scene_container,
                                                transparency_scene_container,
                                                labelmap_scene_container,
                                                mask_overlay_scene_container,
                                                slice_numbers,
                                                fillvalue=tuple()))):

        # Compute the mosaic cell position
        top_pos, left_pos = compute_cell_topleft_pos(idx, cols,
                                                     offset_h, offset_v)

        # Convert the scene data to grayscale and adjust for handling with
        # Pillow
        _img_arr = any2grayscale(img_arr)

        _trans_arr = None
        if len(trans_arr):
            _trans_arr = any2grayscale(trans_arr)

        _labelmap_arr = None
        if len(labelmap_arr):
            _labelmap_arr = any2grayscale(labelmap_arr)

        _mask_overlay_arr = None
        if len(mask_overlay_arr):
            _mask_overlay_arr = mask_overlay_arr

        # Draw the image (and labelmap overlay, if any) in the cell
        compose_image(_img_arr, (cell_width, cell_height), slice_number,
                      corner_position=(left_pos, top_pos),
                      transparency_scene=_trans_arr,
                      labelmap_scene=_labelmap_arr,
                      labelmap_overlay_alpha=labelmap_overlay_alpha,
                      mask_overlay_scene=_mask_overlay_arr,
                      mask_overlay_alpha=mask_overlay_alpha,
                      mask_overlay_color=mask_overlay_color,
                      vol_cmap_name=vol_cmap_name,
                      labelmap_cmap_name=labelmap_cmap_name,
                      display_slice_number=display_slice_number,
                      display_lr=display_lr,
                      lr_labels=lr_labels,
                      canvas=mosaic)

    return mosaic
