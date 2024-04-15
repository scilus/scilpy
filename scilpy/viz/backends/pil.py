# -*- coding: utf-8 -*-

import logging
from matplotlib.font_manager import findfont
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scilpy.viz.color import generate_n_colors


def any2grayscale(array_2d):
    """
    Convert an array to `uint8` grayscale so that it can be appropriately
    handled by `PIL`. The array is normalized to the range [0, 1] before
    the conversion takes place.

    Parameters
    ----------
    array_2d : ndarray
        Data in any range and datatype.

    Returns
    -------
        Grayscale `unit8` data in [0, 255] range.
    """

    # Normalize the data to the range [0, 1]
    def _normalized():
        return (array_2d - np.min(array_2d)) / (
            np.max(array_2d) - np.min(array_2d))

    # Convert from RGB to grayscale
    _gray = Image.fromarray(np.uint8(_normalized() * 255)).convert("L")

    # Relocate overflow values to the dynamic range
    return np.array(_gray).astype("uint8")


def create_image_from_2d_array(array_2d, size, mode=None,
                               resampling=Image.LANCZOS,
                               pixel_dtype=np.uint8):
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
    pixel_dtype : type
        Pixel data type for PIL image. The input array will be cast to this
        type before creating the image using the `Image.fromarray` method.

    Returns
    -------
    image : PIL.Image
        Image.
    """

    # Need to flip the array due to some bug in the FURY image buffer.
    # Might be solved in newer versions of the package.
    return Image.fromarray(array_2d.astype(pixel_dtype), mode=mode) \
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
    greater_threshold: Any
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


def fetch_truetype_font(fontconfig, size, fontface_index=0, encoding="unic",
                        use_default_if_not_found=True):
    """
    Fetch a truetype font using either a `fontconfig pattern` or a
    FontProperties object (see `Matplotlib.font_manager.findfont`_). For
    all other parameters see the `PIL.ImageFont.truetype` method.

    Parameters
    ----------
    fontconfig : str or FontProperties
        Font configuration.
    size : int
        Font size.
    fontface_index : int
        Font face index.
    encoding : str
        Font encoding.
    use_default_if_not_found : bool

    Returns
    -------
    font : ImageFont
        Font.

    .. _Matplotlib.font_manager.findfont:
        https://matplotlib.org/stable/api/font_manager_api.html#matplotlib.font_manager.findfont # noqa
    .. _fontconfig pattern:
        https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
    """

    try:
        font_path = findfont(fontconfig, fallback_to_default=False)
        return ImageFont.truetype(font_path, size, fontface_index, encoding)
    except Exception as e:
        if use_default_if_not_found:
            logging.info(f'Font {fontconfig} was not found. Default font '
                         f'will be used.')
            return ImageFont.load_default(size)
        else:
            raise e


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
    lr_labels : list
        Left/right labels.
    """
    font = fetch_truetype_font("freesans", max(1, image.width // 10))
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
                              image_alpha=1.0,
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
    """

    image = create_image_from_2d_array(array_2d, size, "RGB")

    _transparency = None
    if transparency is not None:
        _transparency = create_image_from_2d_array(transparency * image_alpha,
                                                   size, "RGB", Image.NEAREST)
        _transparency = _transparency.convert("L")
    else:
        _transparency = create_image_from_2d_array(
            np.ones(array_2d.shape[:2]) * image_alpha * 255.,
            size, "L", Image.NEAREST)

    canvas.paste(image, (left_position, top_position), mask=_transparency)

    # Draw the labelmap overlay image if any
    if labelmap_overlay is not None:
        labelmap = create_image_from_2d_array(labelmap_overlay, size, "RGB")
        # Create transparency mask over the labelmap overlay image
        label_mask = np.any(labelmap_overlay > 0, -1) * labelmap_overlay_alpha
        label_transparency = create_image_from_2d_array(label_mask * 255.,
                                                        size, "L",
                                                        Image.NEAREST)

        canvas.paste(labelmap, (left_position, top_position),
                     mask=label_transparency)

    # Draw the mask overlay image if any
    if overlays is not None:
        if overlays_colors is None:
            # Get a list of distinguishable colors if None are supplied
            overlays_colors = generate_n_colors(len(overlays))

        for img, color in zip(overlays, overlays_colors):
            overlay = create_image_from_2d_array(img * color, size, "RGB")

            # Create transparency mask over the mask overlay image
            overlay_transparency = create_image_from_2d_array(
                img * overlays_alpha, size).convert("L")

            canvas.paste(overlay, (left_position, top_position),
                         mask=overlay_transparency)

    if peak_overlay is not None:
        for img in peak_overlay:
            overlay = create_image_from_2d_array(img * 255, size, "RGB")

            # Create transparency mask over the mask overlay image
            overlay_transparency = create_image_from_2d_array(
                img * peak_overlay_alpha, size).convert("L")

            canvas.paste(overlay, (left_position, top_position),
                         mask=overlay_transparency)
