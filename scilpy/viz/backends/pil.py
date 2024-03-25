# -*- coding: utf-8 -*-

from matplotlib.font_manager import findfont
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scilpy.viz.color import generate_n_colors


def any2grayscale(array_2d):
    """
    Convert a [0, 1] bounded array to `uint8` grayscale so that it can be
    appropriately handled by `PIL`. Threshold will be applied to any data
    that overflows a 8bit unsigned container.

    Parameters
    ----------
    array_2d : ndarray
        Data in [0, 1] range.

    Returns
    -------
        Grayscale `unit8` data in [0, 255] range.
    """

    if array_2d.max() > 1:
        raise ValueError(
            "Grayscale conversion requires input data to be in range [0, 1]")

    # Convert from RGB to grayscale
    # TODO : PIL float images, can the uint8 conversion go ?
    _gray = Image.fromarray(np.uint8(array_2d * 255)).convert("L")

    # Relocate overflow values to the dynamic range
    return np.array(_gray * 255).astype("uint8")


def create_image_from_2d_array(array_2d, size, mode=None,
                               resampling=Image.LANCZOS):
    """
    Create a `PIL.Image` from the 2d array data
    (in range [0, 255], if no colormap provided).

    Parameters
    ----------
    array_2d : ndarray
        2d array data.
    size : array-like
        Image size (pixels) (width, height).
    mode : str, optional
        Type and depth of a pixel in the `Pillow` image.
    resampling : Literal, optional
        Resampling method to use when resizing the `Pillow` image.

    Returns
    -------
    image : PIL.Image
        Image.
    """

    # TODO : Need to flip the array due to some bug in the FURY image buffer.
    # Might be solved in newer versions of the package.
    return Image.fromarray(array_2d, mode=mode) \
        .transpose(Image.FLIP_TOP_BOTTOM) \
        .resize(size, resampling)


def create_mask_from_2d_array(array_2d, size, greater_threshold=0):
    """
    Create a binary `PIL.Image` from the 2d array data.

    Parameters
    ----------
    array_2d : ndarray
        2d scene data.
    size : array-like
        Image size (pixels) (width, height).
    greater_threshold: Any, optional
        Threshold to use to binarize the data.
        Type must abide with the array dtype

    Returns
    -------
    image : PIL.Image
        Image.
    """

    _bin_arr = array_2d > greater_threshold
    return create_image_from_2d_array(any2grayscale(_bin_arr), size)


def compute_canvas_size(rows, columns, cell_width, cell_height,
                        width_overlap, height_overlap):
    """
    Compute the size of a canvas with the given number of rows
    and columns, and the requested cell size and overlap values.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    cell_width : int
        Cell width (pixels).
    cell_height : int
        Cell height (pixels).
    width_overlap : int
        Overlap on the image width (pixels).
    height_overlap : int
        Overlap on the image height (pixels).

    Returns
    -------
    size: tuple
        (width, height) of the canvas.
    """

    def _compute_canvas_length(line_count, cell_length, overlap):
        return (line_count - 1) * (cell_length - overlap) + cell_length

    return _compute_canvas_length(columns, cell_width, width_overlap), \
        _compute_canvas_length(rows, cell_height, height_overlap)


def create_canvas(cell_width, cell_height, rows, columns,
                  overlap_horiz, overlap_vert):
    """
    Create a canvas for given number of rows and columns,
    and the requested cell size and overlap values.

    Parameters
    ----------
    cell_width : int
        Cell width (pixels).
    cell_height : int
        Cell height (pixels).
    rows : int
        Row count.
    columns : int
        Column count.
    overlap_horiz : int
        Horizontal overlap (pixels).
    overlap_vert : int
        Vertical overlap (pixels).

    Returns
    -------
    canvas : PIL.Image
        Initialized canvas.
    """

    width, height = compute_canvas_size(rows, columns, cell_width, cell_height,
                                        overlap_horiz, overlap_vert)

    return Image.new("RGBA", (width, height), (0, 0, 0, 0))


def annotate_image(image, slice_number, display_slice_number,
                   display_lr, lr_labels=["L", "R"]):
    """
    Annotate an image with slice number and left/right labels.

    Parameters
    ----------
    image : PIL.Image
        Image to annotate.
    slice_number : int
        Slice number.
    display_slice_number : bool
        Display the slice number in the upper left corner.
    display_lr : int
        Display the left/right labels in the middle of the image. If
        negative, the labels are inverted.
    lr_labels : list, optional
        Left/right labels.
    """
    font_size = image.width // 10
    font = ImageFont.truetype(findfont("freesans", fontext="ttf"), font_size)

    stroke, padding = max(image.width // 200, 1), image.width // 100
    width, height = image.width, image.height
    image = ImageDraw.Draw(image)

    if display_slice_number:
        image.text((padding, padding), "{}".format(slice_number),
                   (255, 255, 255), font=font,
                   stroke_width=stroke, stroke_fill=(0, 0, 0))

    if display_lr:
        l_text, r_text = lr_labels
        if display_lr < 0:
            l_text, r_text = r_text, l_text

        image.text((padding, height // 2), l_text, (255, 255, 255),
                   font=font, anchor="lm",
                   stroke_width=stroke, stroke_fill=(0, 0, 0))

        image.text((width - padding, height // 2),
                   r_text, (255, 255, 255),
                   font=font, anchor="rm",
                   stroke_width=stroke, stroke_fill=(0, 0, 0))


def draw_2d_array_at_position(canvas, array_2d, size,
                              left_position, top_position,
                              transparency=None,
                              labelmap_overlay=None,
                              labelmap_overlay_alpha=0.7,
                              overlays=None,
                              overlays_alpha=0.7,
                              overlays_colors=None,
                              peak_overlay=None,
                              peak_overlay_alpha=0.7):
    """
    Draw a 2d array in the given target image at the specified position.

    Parameters
    ----------
    canvas : PIL.Image
        Target image.
    array_2d : ndarray
        2d array data to be drawn.
    size : array-like
        Image size (pixels) (width, height).
    left_position : int
        Left position (pixels).
    top_position : int
        Top position (pixels).
    transparency : ndarray, optional
        Transparency mask.
    labelmap_overlay : ndarray
        Labelmap overlay scene data to be drawn.
    labelmap_overlay_alpha : float
        Alpha value for labelmap overlay in range [0, 1].
    overlays : ndarray
        Overlays scene data to be drawn.
    overlays_alpha : float
        Alpha value for the overlays in range [0, 1].
    overlays_color : list, optional
        Color for the overlays as a list of 3 integers in range [0, 255].
    peaks_overlay : ndarray
        Peaks overlay scene data to be drawn.
    peaks_overlay_alpha : float
        Alpha value for peaks overlay in range [0, 1].
    vol_lut : function, optional
        Lookup table (colormap) function for the image scene data.
    labelmap_lut_table : function, optional
        Lookup table (colormap) function for the labelmap overlay scene data.
    """

    image = create_image_from_2d_array(array_2d, size, "RGB")

    _transparency = None
    if transparency is not None:
        _transparency = create_image_from_2d_array(transparency, size,
                                                   "RGB", Image.NEAREST)
        _transparency = _transparency.convert("L")

    canvas.paste(image, (left_position, top_position), mask=_transparency)

    # Draw the labelmap overlay image if any
    if labelmap_overlay is not None:
        labelmap = create_image_from_2d_array(labelmap_overlay, size, "RGB")
        # Create transparency mask over the labelmap overlay image
        label_mask = np.any(labelmap_overlay > 0, -1) * labelmap_overlay_alpha
        label_transparency = create_image_from_2d_array(
            (label_mask * 255.).astype(np.uint8), size, "L", Image.NEAREST)

        canvas.paste(labelmap, (left_position, top_position),
                     mask=label_transparency)

    # Draw the mask overlay image if any
    if overlays is not None:
        if overlays_colors is None:
            # Get a list of distinguishable colors if None are supplied
            # TODO : Muddles PIL with fury. Maybe another way to get colors
            overlays_colors = generate_n_colors(len(overlays))

        for img, color in zip(overlays, overlays_colors):
            overlay = create_image_from_2d_array(
                (img * color).astype(np.uint8), size, "RGB")

            # Create transparency mask over the mask overlay image
            overlay_transparency = create_image_from_2d_array(
                (img * overlays_alpha).astype(np.uint8), size).convert("L")

            canvas.paste(overlay, (left_position, top_position),
                         mask=overlay_transparency)

    if peak_overlay is not None:
        for img in peak_overlay:
            overlay = create_image_from_2d_array(
                (img * 255).astype(np.uint8), size, "RGB")

            # Create transparency mask over the mask overlay image
            overlay_transparency = create_image_from_2d_array(
                (img * peak_overlay_alpha).astype(np.uint8), size).convert("L")

            canvas.paste(overlay, (left_position, top_position),
                         mask=overlay_transparency)
