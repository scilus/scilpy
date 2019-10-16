#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import reduce
import itertools

from dipy.tracking.streamline import transform_streamlines
import numpy as np
from scipy import ndimage


MIN_NB_POINTS = 10
KEY_INDEX = np.concatenate((range(5), range(-1, -6, -1)))


def get_streamline_key(streamline, precision=None):
    # Use just a few data points as hash key. I could use all the data of
    # the streamlines, but then the complexity grows with the number of
    # points.
    if len(streamline) < MIN_NB_POINTS:
        key = streamline.copy()
    else:
        key = streamline[KEY_INDEX].copy()

    if precision is not None:
        key = np.round(key, precision)

    key.flags.writeable = False

    return key.data.tobytes()


def hash_streamlines(streamlines, start_index=0, precision=None):
    """Produces a dict from streamlines

    Produces a dict from streamlines by using the points as keys and the
    indices of the streamlines as values.

    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines used to produce the dict.
    start_index: int, optional
        The index of the first streamline. 0 by default.
    precision: int, optional
        The number of decimals to keep when hashing the points of the
        streamlines. Allows a soft comparison of streamlines. If None, no
        rounding is performed.

    Returns
    -------
    A dict where the keys are streamline points and the values are indices
    starting at start_index.

    """

    keys = [get_streamline_key(s, precision) for s in streamlines]
    return {k: i for i, k in enumerate(keys, start_index)}


def intersection(left, right):
    """Intersection of two streamlines dict (see hash_streamlines)"""
    return {k: v for k, v in left.items() if k in right}


def subtraction(left, right):
    """Subtraction of two streamlines dict (see hash_streamlines)"""
    return {k: v for k, v in left.items() if k not in right}


def union(left, right):
    """Union of two streamlines dict (see hash_streamlines)"""

    # In python 3 : return {**left, **right}
    result = left.copy()
    result.update(right)
    return result


def perform_streamlines_operation(operation, streamlines, precision=None):
    """Peforms an operation on a list of list of streamlines

    Given a list of list of streamlines, this function applies the operation
    to the first two lists of streamlines. The result in then used recursively
    with the third, fourth, etc. lists of streamlines.

    A valid operation is any function that takes two streamlines dict as input
    and produces a new streamlines dict (see hash_streamlines). Union,
    subtraction, and intersection are valid examples of operations.

    Parameters
    ----------
    operation: callable
        A callable that takes two streamlines dicts as inputs and preduces a
        new streamline dict.
    streamlines: list of list of streamlines
        The streamlines used in the operation.
    precision: int, optional
        The number of decimals to keep when hashing the points of the
        streamlines. Allows a soft comparison of streamlines. If None, no
        rounding is performed.

    Returns
    -------
    streamlines: list of `nib.streamline.Streamlines`
        The streamlines obtained after performing the operation on all the
        input streamlines.
    indices: list
        The indices of the streamlines that are used in the output.

    """

    # Hash the streamlines using the desired precision.
    indices = np.cumsum([0] + [len(s) for s in streamlines[:-1]])
    hashes = [hash_streamlines(s, i, precision) for
              s, i in zip(streamlines, indices)]

    # Perform the operation on the hashes and get the output streamlines.
    to_keep = reduce(operation, hashes)
    all_streamlines = list(itertools.chain(*streamlines))
    indices = sorted(to_keep.values())
    streamlines = [all_streamlines[i] for i in indices]
    return streamlines, indices


def warp_tractogram(streamlines, transfo, deformation_data, source):
    """
    Warp tractogram using a deformation map.
    Support Ants and Dipy deformation map.
    Apply warp in-place

    Parameters
    ----------
    streamlines: list or ArraySequence
        Streamlines as loaded by the nibabel API (RASMM)
    transfo: numpy.ndarray
        Transformation matrix to bring streamlines from RASMM to Voxel space
    deformation_data: numpy.ndarray
        4D numpy array containing a 3D displacement vector in each voxel
    source: str
        Source of the deformation map [ants, dipy]
    """

    if source == 'ants':
        flip = [-1, -1, 1]
    elif source == 'dipy':
        flip = [1, 1, 1]

    # Because of duplication, an iteration over chunks of points is necessary
    # for a big dataset (especially if not compressed)
    nb_points = len(streamlines._data)
    current_position = 0
    chunk_size = 1000000
    nb_iteration = int(np.ceil(nb_points/chunk_size))
    inv_transfo = np.linalg.inv(transfo)

    while nb_iteration > 0:
        max_position = min(current_position + chunk_size, nb_points)
        streamline = streamlines._data[current_position:max_position]

        # To access the deformation information, we need to go in voxel space
        streamline_vox = transform_streamlines(streamline,
                                               inv_transfo)

        current_streamline_vox = np.array(streamline_vox).T
        current_streamline_vox_list = current_streamline_vox.tolist()

        x_def = ndimage.map_coordinates(deformation_data[..., 0],
                                        current_streamline_vox_list, order=1)
        y_def = ndimage.map_coordinates(deformation_data[..., 1],
                                        current_streamline_vox_list, order=1)
        z_def = ndimage.map_coordinates(deformation_data[..., 2],
                                        current_streamline_vox_list, order=1)

        # ITK is in LPS and nibabel is in RAS, a flip is necessary for ANTs
        final_streamline = np.array([flip[0]*x_def,
                                     flip[1]*y_def,
                                     flip[2]*z_def])

        # The deformation obtained is in worldSpace
        if source == 'ants':
            final_streamline += np.array(streamline).T
        elif source == 'dipy':
            final_streamline += current_streamline_vox
            # The tractogram need to be brought back in world space to be saved
            final_streamline = transform_streamlines(final_streamline,
                                                     transfo)

        streamlines._data[current_position:max_position] \
            = final_streamline.T
        current_position = max_position
        nb_iteration -= 1
