# -*- coding: utf-8 -*-

from functools import reduce
import itertools
import logging

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamlinespeed import compress_streamlines
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from scipy.ndimage import map_coordinates

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


def difference(left, right):
    """Difference of two streamlines dict (see hash_streamlines)"""
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
    difference, and intersection are valid examples of operations.

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


def warp_streamlines(sft, deformation_data, source='ants'):
    """ Warp tractogram using a deformation map. Apply warp in-place.
    Support Ants and Dipy deformation map.

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
    sft.to_rasmm()
    sft.to_center()
    streamlines = sft.streamlines
    transfo = sft.affine
    if source == 'ants':
        flip = [-1, -1, 1]
    elif source == 'dipy':
        flip = [1, 1, 1]

    # Because of duplication, an iteration over chunks of points is necessary
    # for a big dataset (especially if not compressed)
    streamlines = ArraySequence(streamlines)
    nb_points = len(streamlines._data)
    cur_position = 0
    chunk_size = 1000000
    nb_iteration = int(np.ceil(nb_points/chunk_size))
    inv_transfo = np.linalg.inv(transfo)

    while nb_iteration > 0:
        max_position = min(cur_position + chunk_size, nb_points)
        points = streamlines._data[cur_position:max_position]

        # To access the deformation information, we need to go in voxel space
        # No need for corner shift since we are doing interpolation
        cur_points_vox = np.array(transform_streamlines(points,
                                                        inv_transfo)).T

        x_def = map_coordinates(deformation_data[..., 0],
                                cur_points_vox.tolist(), order=1)
        y_def = map_coordinates(deformation_data[..., 1],
                                cur_points_vox.tolist(), order=1)
        z_def = map_coordinates(deformation_data[..., 2],
                                cur_points_vox.tolist(), order=1)

        # ITK is in LPS and nibabel is in RAS, a flip is necessary for ANTs
        final_points = np.array([flip[0]*x_def, flip[1]*y_def, flip[2]*z_def])

        # The Ants deformation is relative to world space
        if source == 'ants':
            final_points += np.array(points).T
        # Dipy transformation is relative to vox space
        elif source == 'dipy':
            final_points += cur_points_vox
            transform_streamlines(final_points, transfo, in_place=True)
        streamlines._data[cur_position:max_position] = final_points.T
        cur_position = max_position
        nb_iteration -= 1

        return streamlines


def filter_tractogram_data(tractogram, streamline_ids):
    """ Filter tractogram according to streamline ids and keep the data

    Parameters:
    -----------
    tractogram: StatefulTractogram
        Tractogram containing the data to be filtered
    streamline_ids: array_like
        List of streamline ids the data corresponds to

    Returns:
    --------
    new_tractogram: Tractogram or StatefulTractogram
        Returns a new tractogram with only the selected streamlines
        and data
    """

    streamline_ids = np.asarray(streamline_ids, dtype=np.int)

    assert np.all(
        np.in1d(streamline_ids, np.arange(len(tractogram.streamlines)))
    ), "Received ids outside of streamline range"

    new_streamlines = tractogram.streamlines[streamline_ids]
    new_data_per_streamline = tractogram.data_per_streamline[streamline_ids]
    new_data_per_point = tractogram.data_per_point[streamline_ids]

    # Could have been nice to deepcopy the tractogram modify the attributes in
    # place instead of creating a new one, but tractograms cant be subsampled
    # if they have data

    return StatefulTractogram.from_sft(
        new_streamlines,
        tractogram,
        data_per_point=new_data_per_point,
        data_per_streamline=new_data_per_streamline)


def compress_sft(sft, tol_error=0.01):
    """ Compress a stateful tractogram. Uses Dipy's compress_streamlines, but
    deals with space better.

    Dipy's description:
    The compression consists in merging consecutive segments that are
    nearly collinear. The merging is achieved by removing the point the two
    segments have in common.

    The linearization process [Presseau15]_ ensures that every point being
    removed are within a certain margin (in mm) of the resulting streamline.
    Recommendations for setting this margin can be found in [Presseau15]_
    (in which they called it tolerance error).

    The compression also ensures that two consecutive points won't be too far
    from each other (precisely less or equal than `max_segment_length`mm).
    This is a tradeoff to speed up the linearization process [Rheault15]_. A low
    value will result in a faster linearization but low compression, whereas
    a high value will result in a slower linearization but high compression.

    [Presseau C. et al., A new compression format for fiber tracking datasets,
    NeuroImage, no 109, 73-83, 2015.]

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to compress.
    tol_error: float (optional)
        Tolerance error in mm (default: 0.01). A rule of thumb is to set it
        to 0.01mm for deterministic streamlines and 0.1mm for probabilitic
        streamlines.

    Returns
    -------
    compressed_sft : StatefulTractogram
    """
    # Go to world space
    orig_space = sft.space
    sft.to_rasmm()

    # Compress streamlines
    compressed_streamlines = compress_streamlines(sft.streamlines,
                                                  tol_error=tol_error)
    if sft.data_per_point is not None:
        logging.warning("Initial stateful tractogram contained data_per_point. "
                        "This information will not be carried in the final"
                        "tractogram.")

    compressed_sft = StatefulTractogram.from_sft(
        compressed_streamlines, sft,
        data_per_streamline=sft.data_per_streamline)

    # Return to original space
    compressed_sft.to_space(orig_space)

    return compressed_sft
