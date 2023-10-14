# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scilpy.viz.color import generate_n_colors, get_colormap



def rgb2gray4pil(rgb_arr):
    """Convert an RGB array to grayscale and convert to `uint8` so that it can
    be appropriately handled by `PIL`.

    Parameters
    ----------
    rgb_arr : ndarray
        RGB value data.

    Returns
    -------
    Grayscale `unit8` data.
    """

    def _rgb2gray(rgb):
        img = Image.fromarray(np.uint8(rgb * 255)).convert("L")
        return np.array(img)

    # Convert from RGB to grayscale
    gray_arr = _rgb2gray(rgb_arr)

    # Relocate overflow values to the dynamic range
    return (gray_arr * 255).astype("uint8")


def create_image_from_2d_scene(scene_2d, size, mode=None, cmap_name=None):
    """Create a `PIL.Image` from the 2d scene data.

    Parameters
    ----------
    array_2d : ndarray
        2d scene data.
    size : array-like
        Image size (pixels) (width, height).
    mode : str, optional
        Type and depth of a pixel in the `Pillow` image.
    cmap_name : str, optional
        Colormap name.

    Returns
    -------
    image : PIL.Image
        Image.
    """

    if cmap_name:
        # Apply the colormap
        cmap = get_colormap(cmap_name)
        # data returned by cmap is normalized to the [0,1] range: scale to the
        # [0, 255] range and convert to uint8 for Pillow
        scene_2d = (cmap(scene_2d) * 255).astype("uint8")

    # Need to flip the array due to some bug in the FURY image buffer. Might be
    # solved in newer versions of the package.
    image = Image.fromarray(
        scene_2d, mode=mode).transpose(Image.FLIP_TOP_BOTTOM)

    return image.resize(size, Image.ANTIALIAS)


def create_mask_from_2d_scene(scene_2d, size):
    """Create a binary `PIL.Image` from the 2d scene data.

    Parameters
    ----------
    scene_2d : ndarray
        2d scene data.
    size : array-like
        Image size (pixels) (width, height).

    Returns
    -------
    image : PIL.Image
        Image.
    """

    _bin_arr = scene_2d > 0
    _bin_arr = rgb2gray4pil(_bin_arr) * 255
    image = create_image_from_2d_scene(_bin_arr, size)

    return image


def compute_canvas_size(
    cell_width,
    cell_height,
    overlap_horiz,
    overlap_vert,
    rows,
    cols,
):
    """Compute the size of a canvas with the given number of rows and columns,
    and the requested cell size and overlap values.

    Parameters
    ----------
    cell_width : int
        Cell width (pixels).
    cell_height : int
        Cell height (pixels).
    overlap_horiz : int
        Horizontal overlap (pixels).
    overlap_vert : int
        Vertical overlap (pixels).
    rows : int
        Row count.
    cols : int
        Column count.
    """

    def _compute_canvas_length(line_count, cell_length, overlap):
        return (line_count - 1) * (cell_length - overlap) + cell_length

    width = _compute_canvas_length(cols, cell_width, overlap_horiz)
    height = _compute_canvas_length(rows, cell_height, overlap_vert)

    return width, height


def create_canvas(
    cell_width,
    cell_height,
    overlap_horiz,
    overlap_vert,
    rows,
    cols,
):
    """Create a canvas for given number of rows and columns, and the requested
     cell size and overlap values.

    Parameters
    ----------
    cell_width : int
        Cell width (pixels).
    cell_height : int
        Cell height (pixels).
    overlap_horiz : int
        Horizontal overlap (pixels).
    overlap_vert : int
        Vertical overlap (pixels).
    rows : int
        Row count.
    cols : int
        Column count.
    """

    width, height = compute_canvas_size(
        cell_width, cell_height, overlap_horiz, overlap_vert, rows, cols
    )
    mosaic = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    return mosaic


def annotate_scene(mosaic, slice_number, display_slice_number, display_lr):
    font_size = mosaic.width // 10
    font = ImageFont.truetype(
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf', font_size)

    stroke, padding = max(mosaic.width // 200, 1), mosaic.width // 100
    img = ImageDraw.Draw(mosaic)

    if display_slice_number:
        img.text(
            (padding, padding), "{}".format(slice_number), (255,255,255),
            font=font, stroke_width=stroke, stroke_fill=(0, 0, 0)
        )

    if display_lr:
        l_text, r_text = "L", "R"
        if display_lr < 0:
            l_text, r_text = r_text, l_text

        img.text(
            (padding, mosaic.height // 2), l_text, (255,255,255),
            font=font, anchor="lm", stroke_width=stroke, stroke_fill=(0, 0, 0)
        )
        img.text(
            (mosaic.width - padding, mosaic.height // 2), r_text, (255,255,255),
            font=font, anchor="rm", stroke_width=stroke, stroke_fill=(0, 0, 0)
        )


def draw_scene_at_pos(
    canvas,
    scene,
    size,
    left_pos,
    top_pos,
    transparency=None,
    labelmap_overlay=None,
    labelmap_overlay_alpha=0.7,
    mask_overlay=None,
    mask_overlay_alpha=0.7,
    mask_overlay_color=None,
    vol_cmap_name=None,
    labelmap_cmap_name=None,
):
    """Draw a scene in the given target image at the specified position.

    Parameters
    ----------
    canvas : PIL.Image
        Target image.
    scene : ndarray
        Scene data to be drawn.
    size : array-like
        Image size (pixels) (width, height).
    left_pos : int
        Left position (pixels).
    top_pos : int
        Top position (pixels).
    transparency : ndarray, optional
        Transparency mask.
    labelmap_overlay : ndarray
        Labelmap overlay scene data to be drawn.
    mask_overlay : ndarray
        Mask overlay scene data to be drawn.
    mask_overlay_alpha : float
        Alpha value for mask overlay in range [0, 1].
    mask_overlay_color : list, optional
        Color for the mask overlay as a list of 3 integers in range [0, 255].
    vol_cmap_name : str, optional
        Colormap name for the image scene data.
    labelmap_cmap_name : str, optional
        Colormap name for the labelmap overlay scene data.
    """

    image = create_image_from_2d_scene(scene, size, cmap_name=vol_cmap_name)

    trans_img = None
    if transparency is not None:
        trans_img = create_image_from_2d_scene(transparency, size, mode="L")

    canvas.paste(image, (left_pos, top_pos), mask=trans_img)

    # Draw the labelmap overlay image if any
    if labelmap_overlay is not None:
        labelmap_img = create_image_from_2d_scene(
            labelmap_overlay, size, cmap_name=labelmap_cmap_name
        )

        # Create transparency mask over the labelmap overlay image
        label_mask = labelmap_overlay > 0
        label_transparency = create_image_from_2d_scene(
            (label_mask * labelmap_overlay_alpha * 255.).astype(np.uint8),
            size).convert("L")

        canvas.paste(labelmap_img, (left_pos, top_pos), mask=label_transparency)

    # Draw the mask overlay image if any
    if mask_overlay is not None:
        if mask_overlay_color is None:
            # Get a list of distinguishable colors if None are supplied
            mask_overlay_color = generate_n_colors(len(mask_overlay))

        for img, color in zip(mask_overlay, mask_overlay_color):
            overlay_img = create_image_from_2d_scene(
                (img * color).astype(np.uint8), size, "RGB")

            # Create transparency mask over the mask overlay image
            overlay_trans = create_image_from_2d_scene(
                (img * mask_overlay_alpha).astype(np.uint8), size).convert("L")

            canvas.paste(overlay_img, (left_pos, top_pos), mask=overlay_trans)