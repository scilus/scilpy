# -*- coding: utf-8 -*-

import numpy as np
from numpy.lib.index_tricks import r_ as row


def voxel_to_world(coord, affine):
    """Takes a n dimensionnal voxel coordinate and returns its 3 first
    coordinates transformed to world space from a given voxel to world affine
    transformation."""

    normalized_coord = row[coord[0:3], 1.0].astype(float)
    world_coord = np.dot(affine, normalized_coord)
    return world_coord[0:3]


def world_to_voxel(coord, affine):
    """Takes a n dimensionnal world coordinate and returns its 3 first
    coordinates transformed to voxel space from a given voxel to world affine
    transformation."""

    normalized_coord = row[coord[0:3], 1.0].astype(float)
    iaffine = np.linalg.inv(affine)
    vox_coord = np.dot(iaffine, normalized_coord)
    vox_coord = np.round(vox_coord).astype(int)
    return vox_coord[0:3]


def str_to_index(axis):
    """
    Convert x y z axis string to 0 1 2 axis index

    Parameters
    ----------
    axis: str
        Axis value (x, y or z)

    Returns
    -------
    index: int or None
        Axis index
    """
    axis = axis.lower()
    axes = {'x': 0, 'y': 1, 'z': 2}

    if axis in axes:
        return axes[axis]

    return None


def is_float(value):
    """Returns True if the argument can be casted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False
