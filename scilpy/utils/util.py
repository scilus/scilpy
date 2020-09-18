# -*- coding: utf-8 -*-

from dipy.io.utils import get_reference_info
import numpy as np
from numpy.lib.index_tricks import r_ as row


def compute_distance_barycenters(ref_1, ref_2, ref_2_transfo):
    """
    Compare the barycenter (center of volume) of two reference object.
    The provided transformation will move the reference #2 and
    return the distance before and after transformation.

    Parameters
    ----------
    ref_1: reference object
        Any type supported by the sft as reference (e.g .nii of .trk).
    ref_2: reference object
        Any type supported by the sft as reference (e.g .nii of .trk).
    ref_2_transfo: np.ndarray
        Transformation that modifies the barycenter of ref_2.
    Returns
    -------
    distance: float or tuple (2,)
        return a tuple containing the distance before and after
        the transformation.
    """
    aff_1, dim_1, _, _ = get_reference_info(ref_1)
    aff_2, dim_2, _, _ = get_reference_info(ref_2)

    barycenter_1 = voxel_to_world(dim_1 / 2.0, aff_1)
    barycenter_2 = voxel_to_world(dim_2 / 2.0, aff_2)
    distance_before = np.linalg.norm(barycenter_1 - barycenter_2)

    normalized_coord = row[barycenter_2[0:3], 1.0].astype(float)
    barycenter_2 = np.dot(ref_2_transfo, normalized_coord)[0:3]

    distance_after = np.linalg.norm(barycenter_1 - barycenter_2)

    return distance_before, distance_after


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
