# -*- coding: utf-8 -*-

import numpy as np

from scilpy.utils.spatial import get_axis_index


def affine_from_offset(orientation, offset):
    """
    Create an affine matrix from a scalar offset in given orientation,
    in RPS coordinates for imaging.

    Parameters
    ----------
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    offset : float
        The offset of the texture image.

    Returns
    -------
    affine : np.ndarray
        The affine transformation.
    """

    offset_flip, ax_idx = [1., -1., 1.], get_axis_index(orientation)
    affine = np.identity(4)
    affine[ax_idx, 3] = offset_flip[ax_idx] * offset
    return affine


def check_mosaic_layout(img_count, rows, cols):
    """
    Check whether a mosaic can be built given the image count and the
    requested number of rows and columns. Raise a `ValueError` if it cannot be
    built.

    Parameters
    ----------
    img_count : int
        Image count to be arranged in the mosaic.
    rows : int
        Row count.
    cols : int
        Column count.
    """

    cell_count = rows * cols

    if img_count < cell_count:
        raise ValueError(
            f"Less slices than cells requested.\nImage count: {img_count}; "
            f"Cell count: {cell_count} (rows: {rows}; cols: {cols}).\n"
            "Please provide an appropriate value for the rows, cols for the "
            "slice count.")
    elif img_count > cell_count:
        raise ValueError(
            f"More slices than cells requested.\nImage count: {img_count}; "
            f"Cell count: {cell_count} (rows: {rows}; cols: {cols}).\n"
            "Please provide an appropriate value for the rows, cols for the "
            "slice count.")


def compute_cell_topleft_pos(idx, cols, offset_h, offset_v):
    """
    Compute the top-left position of a cell to be drawn in a mosaic.

    Parameters
    ----------
    idx : int
       Cell index in the mosaic.
    cols : int
        Column count.
    offset_h :
        Horizontal offset (pixels).
    offset_v :
        Vertical offset (pixels).

    Returns
    -------
    top_pos : int
        Top position (pixels).
    left_pos : int
        Left position (pixels).
    """

    row_idx = int(np.floor(idx / cols))
    top_pos = row_idx * offset_v
    col_idx = idx % cols
    left_pos = col_idx * offset_h

    return top_pos, left_pos
