# -*- coding: utf-8 -*-
from dipy.io.utils import get_reference_info
import nibabel.orientations as ornt
import numpy as np

from numpy.lib.index_tricks import r_ as row


RAS_AXES_NAMES = ["sagittal", "coronal", "axial"]
RAS_AXES_COORDINATES = ["x", "y", "z"]
RAS_AXES_BASIS_VECTORS = ["i", "j", "k"]


def _any2ras_index(axis_index, affine=np.eye(4)):
    """
    Get the index and sign of an axis in RAS from a given index in
    a frame of reference defined by an affine transformation.

    Parameters
    ----------
    axis_index : int
        Index of the axis.
    affine : np.array, optional
        An affine used to compute axis reordering from RAS.

    Returns
    -------
    _ix : int
        Index of the axis in RAS.
    _sgn : str
        Sign of the axis in RAS.
    """
    _ornt = ornt.io_orientation(affine)
    _ix = int(_ornt[axis_index, 0])
    _sgn = "" if _ornt[axis_index, 1] > 0 else "-"
    return _ix, _sgn


def get_axis_name(axis_index, affine=np.eye(4)):
    """
    Get the axis name in :py:const:`~scilpy.utils.spatial.RAS_AXES_NAMES`
    related to a given index.

    Parameters
    ----------
    axis_index : int
        Index of the axis.
    affine : np.array, optional
        An affine used to compute axis reordering from RAS.

    Returns
    -------
    axis_name : str
        Name of the axis.
    """
    _ix, _ = _any2ras_index(axis_index, affine)
    return RAS_AXES_NAMES[_ix]


def get_coordinate_name(axis_index, affine=np.eye(4)):
    """
    Get the signed coordinate in :py:const:`~scilpy.utils.spatial.RAS_AXES_COORDINATES`
    related to a given axis index.

    Parameters
    ----------
    axis_index : int
        Index of the axis.
    affine : np.array, optional
        An affine used to compute axis reordering from RAS.

    Returns
    -------
    coordinate_name : str
        Name of the coordinate suffixed with sign (see RAS_AXES_COORDINATES).
    """ # noqa
    _ix, _sgn = _any2ras_index(axis_index, affine)
    return RAS_AXES_COORDINATES[_ix] + _sgn


def get_basis_vector_name(axis_index, affine=np.eye(4)):
    """
    Get the signed basis vector name in RAS_AXES_BASIS_VECTORS related to a
    given axis index.

    Parameters
    ----------
    axis_index : int
        Index of the axis.
    affine : np.array, optional
        An affine used to compute axis reordering from RAS.

    Returns
    -------
    basis_vector_name : str
        Name of the basis vector suffixed with sign (see
        RAS_AXES_BASIS_VECTORS).
    """
    _ix, _sgn = _any2ras_index(axis_index, affine)
    return RAS_AXES_BASIS_VECTORS[_ix] + _sgn


def get_axis_index(axis, affine=np.eye(4)):
    """
    Get the axis index (or position) in the image from the axis,
    coordinate or basis vector name.

    Parameters
    ----------
    axis : str
        Either an axis name (see RAS_AXES_NAMES), a coordinate name
        (see RAS_AXES_COORDINATES) or a basis vector name
        (see RAS_AXES_BASIS_VECTORS).
    affine : np.array, optional
        An affine used to compute axis reordering from RAS.

    Returns
    -------
    axis_index : int
        Index of the axis.
    """
    _cmp, _ax = None, axis.lower()
    if _ax in RAS_AXES_NAMES:
        _cmp = RAS_AXES_NAMES
    elif _ax in RAS_AXES_COORDINATES:
        _cmp = RAS_AXES_COORDINATES
        _ax = _ax if len(_ax) == 1 else _ax[0]
    elif _ax in RAS_AXES_BASIS_VECTORS:
        _cmp = RAS_AXES_BASIS_VECTORS
        _ax = _ax if len(_ax) == 1 else _ax[0]
    else:
        raise ValueError("Name does not correspond to any axis, coordinate "
                         "or basis vector name : {}".format(axis))

    _ornt = ornt.io_orientation(affine)
    return np.array(_cmp)[_ornt[:, 0].astype(int)].tolist().index(_ax)


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
    """
    Class to store the bounding box of a volume in world space.

    Parameters
    ----------
    minimums: np.ndarray
        Minimums of the bounding box.
    maximums: np.ndarray
        Maximums of the bounding box.
    voxel_size: np.ndarray
        Voxel size of the bounding box.
    """
    def __init__(self, minimums, maximums, voxel_size):
        self.minimums = minimums
        self.maximums = maximums
        self.voxel_size = voxel_size


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
