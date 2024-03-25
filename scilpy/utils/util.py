# -*- coding: utf-8 -*-

import collections.abc

from dipy.io.utils import get_reference_info
from dipy.segment.mask import bounding_box
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


class WorldBoundingBox(object):
    def __init__(self, minimums, maximums, voxel_size):
        self.minimums = minimums
        self.maximums = maximums
        self.voxel_size = voxel_size


def voxel_to_world(coord, affine):
    """
    Takes a n dimensionnal voxel coordinate and returns its 3 first
    coordinates transformed to world space from a given voxel to world affine
    transformation.

    Parameters
    ----------
    coord: np.ndarray
        N-dimensional world coordinate array.
    affine: np.array
        Image affine.

    Returns
    -------
    world_coord: np.ndarray
        Array of world coordinates.
    """

    normalized_coord = row[coord[0:3], 1.0].astype(float)
    world_coord = np.dot(affine, normalized_coord)
    return world_coord[0:3]


def world_to_voxel(coord, affine):
    """
    Takes a n dimensionnal world coordinate and returns its 3 first
    coordinates transformed to voxel space from a given voxel to world affine
    transformation.

    Parameters
    ----------
    coord: np.ndarray
        N-dimensional world coordinate array.
    affine: np.array
        Image affine.

    Returns
    -------
    vox_coord: np.ndarray
        Array of voxel coordinates.
    """

    normalized_coord = row[coord[0:3], 1.0].astype(float)
    iaffine = np.linalg.inv(affine)
    vox_coord = np.dot(iaffine, normalized_coord)
    vox_coord = np.round(vox_coord).astype(int)
    return vox_coord[0:3]


def compute_nifti_bounding_box(img):
    """
    Finds bounding box from data and transforms it in world space for use
    on data with different attributes like voxel size.

    Parameters
    ----------
    img: nib.Nifti1Image
        Input image file.

    Returns
    -------
    wbbox: WorldBoundingBox Object
        Bounding box in world space.
    """
    data = img.get_fdata(dtype=np.float32, caching='unchanged')
    affine = img.affine
    voxel_size = img.header.get_zooms()[0:3]

    voxel_bb_mins, voxel_bb_maxs = bounding_box(data)

    world_bb_mins = voxel_to_world(voxel_bb_mins, affine)
    world_bb_maxs = voxel_to_world(voxel_bb_maxs, affine)
    wbbox = WorldBoundingBox(world_bb_mins, world_bb_maxs, voxel_size)

    return wbbox


def is_float(value):
    """Returns True if the argument can be casted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def recursive_update(d, u, from_existing=False):
    """Harmonize a dictionary to garantee all keys exists at all sub-levels."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            if k not in d and from_existing:
                d[k] = u[k]
            else:
                d[k] = recursive_update(d.get(k, {}), v,
                                        from_existing=from_existing)
        else:
            if not from_existing:
                d[k] = float('nan')
            elif k not in d:
                d[k] = float('nan')
    return d


def recursive_print(data):
    """Print the keys of all layers. Dictionary must be harmonized first."""
    if isinstance(data, collections.abc.Mapping):
        print(list(data.keys()))
        recursive_print(data[list(data.keys())[0]])
    else:
        return


def rotation_around_vector_matrix(vec, theta):
    """ Rotation matrix around a 3D vector by an angle theta.
    From https://stackoverflow.com/questions/6802577/rotation-of-3d-vector

    Parameters
    ----------
    vec: ndarray (3,)
        The vector to rotate around.
    theta: float
        The angle of rotation in radians.

    Returns
    -------
    rot: ndarray (3, 3)
        The rotation matrix.
    """

    vec = vec / np.linalg.norm(vec)
    x, y, z = vec
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c + x**2 * (1 - c),
                      x * y * (1 - c) - z * s,
                      x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s,
                         c + y**2 * (1 - c),
                         y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s,
                         z * y * (1 - c) + x * s,
                         c + z**2 * (1 - c)]])
