# -*- coding: utf-8 -*-

"""
This module regroups small operations and util functions for tractogram.
Meaning that these operations are applied on the streamlines as wholes (ex,
registration, suffling, etc), not on each point of the streamlines separately /
individually. See scilpy.tractograms.streamline_operations.py for the latter.
"""

from functools import reduce
import itertools
import logging
import random

from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import get_reference_info, is_header_compatible
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamlinespeed import compress_streamlines
from nibabel.streamlines import TrkFile, TckFile
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from scilpy.tractograms.streamline_operations import smooth_line_gaussian, \
    resample_streamlines_step_size, parallel_transport_streamline, \
    cut_invalid_streamlines

MIN_NB_POINTS = 10
KEY_INDEX = np.concatenate((range(5), range(-1, -6, -1)))


def shuffle_streamlines(sft, rng_seed=None):
    indices = np.arange(len(sft.streamlines))
    random.shuffle(indices, random=rng_seed)

    streamlines = sft.streamlines[indices]
    data_per_streamline = sft.data_per_streamline[indices]
    data_per_point = sft.data_per_point[indices]

    shuffled_sft = StatefulTractogram.from_sft(
        streamlines, sft,
        data_per_streamline=data_per_streamline,
        data_per_point=data_per_point)
    return shuffled_sft


def get_axis_flip_vector(flip_axes):
    """
    Create a flip vector from a list of axes.

    Parameters
    ----------
    flip_axes: list
        List of axes you want to flip

    Returns
    -------
    shift_vector: list[3,]
        Vector with flipped axes
    """
    flip_vector = np.ones(3)
    if 'x' in flip_axes:
        flip_vector[0] = -1.0
    if 'y' in flip_axes:
        flip_vector[1] = -1.0
    if 'z' in flip_axes:
        flip_vector[2] = -1.0

    return flip_vector


def _get_shift_vector(sft):
    dims = sft.space_attributes[1]
    shift_vector = -1.0 * (np.array(dims) / 2.0)

    return shift_vector


def flip_sft(sft, flip_axes):
    """
    Parameters
    ----------
    sft: StatefulTractogram
    flip_axes: List[str]
        The list of axes to flip. Ex: ['x', 'y', 'z']. The axes correspond to
        the coordinates of the sft as it is stored in memory, not to axes
        of the image.

    Returns
    -------
    flipped_sft: StatefulTractogram
    """
    if len(flip_axes) == 0:
        # Could return sft. But creating new SFT (or deep copy).
        flipped_streamlines = sft.streamlines
    else:
        flip_vector = get_axis_flip_vector(flip_axes)
        shift_vector = _get_shift_vector(sft)

        flipped_streamlines = []

        for streamline in sft.streamlines:
            mod_streamline = streamline + shift_vector
            mod_streamline *= flip_vector
            mod_streamline -= shift_vector
            flipped_streamlines.append(mod_streamline)

    new_sft = StatefulTractogram.from_sft(
        flipped_streamlines, sft,
        data_per_point=sft.data_per_point,
        data_per_streamline=sft.data_per_streamline)

    return new_sft


def _get_streamline_key(streamline, precision=None):
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


def _hash_streamlines(streamlines, start_index=0, precision=None):
    """
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
    keys = [_get_streamline_key(s, precision) for s in streamlines]

    return {k: i for i, k in enumerate(keys, start_index)}


def intersection(left, right):
    """Intersection of two streamlines dict (see hash_streamlines)"""
    return {k: v for k, v in left.items() if k in right}


def difference(left, right):
    """Difference of two streamlines dict (see hash_streamlines)"""
    return {k: v for k, v in left.items() if k not in right}


def union(left, right):
    """Union of two streamlines dict (see hash_streamlines)"""
    return {**left, **right}


def perform_tractogram_operation_on_sft(op_name, sft_list, precision,
                                        no_metadata, fake_metadata):
    """Peforms an operation on a list of tractograms.

    Parameters
    ----------
    op_name: str
        A callable that takes two streamlines dicts as inputs and preduces a
        new streamline dict.
    sft_list: list[StatefulTractogram]
        The streamlines used in the operation.
    precision: int, optional
        The number of decimals to keep when hashing the points of the
        streamlines. Allows a soft comparison of streamlines. If None, no
        rounding is performed.
    no_metadata: bool
        If true, remove all metadata.
    fake_metadata: bool
        If true, fake metadata for SFTs that do not contain the keys available
        in other SFTs.

    Returns
    -------
    sft: StatefulTractogram
        The final SFT
    """
    # Performing operation
    streamlines_list = [sft.streamlines if sft is not None else []
                        for sft in sft_list]
    _, indices = perform_tractogram_operation_on_lines(
        OPERATIONS[op_name], streamlines_list, precision=precision)

    # Current error in dipy prevents concatenation with empty SFT
    # (see PR here to fix: https://github.com/dipy/dipy/pull/2864)
    # Returning empty sft now if that is the case.
    if len(indices) == 0:
        empty_sft = sft_list[0]
        empty_sft.streamlines = []
        return empty_sft, indices

    # Concatenating only the necessary streamlines, with the metadata
    indices_per_sft = []
    streamlines_len_cumsum = [len(sft) for sft in sft_list]
    start = 0
    for nb in streamlines_len_cumsum:
        end = start + nb
        # Switch to int32 for json
        indices_per_sft.append([int(i - start)
                                for i in indices if start <= i < end])
        start = end

    sft_list = [sft[indices_per_sft[i]] for i, sft in enumerate(sft_list)
                if len(indices_per_sft[i]) > 0]

    new_sft = concatenate_sft(sft_list, no_metadata, fake_metadata)
    return new_sft, indices_per_sft


def perform_tractogram_operation_on_lines(operation, streamlines,
                                          precision=None):
    """Peforms an operation on a list of list of streamlines.

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
    indices: np.ndarray
        The indices of the streamlines that are used in the output.
    """
    if 'robust' in operation.__name__:
        if precision is None:
            precision = 3
        return operation(streamlines, precision)
    else:
        # Hash the streamlines using the desired precision.
        indices = np.cumsum([0] + [len(s) for s in streamlines[:-1]])
        hashes = [_hash_streamlines(s, i, precision) for
                  s, i in zip(streamlines, indices)]

        # Perform the operation on the hashes and get the output streamlines.
        to_keep = reduce(operation, hashes)
        all_streamlines = list(itertools.chain(*streamlines))
        indices = np.array(sorted(to_keep.values())).astype(np.uint32)
        streamlines = [all_streamlines[i] for i in indices]
    return streamlines, indices


def intersection_robust(streamlines_list, precision=3):
    """ Intersection of a list of StatefulTractogram """
    if not isinstance(streamlines_list, list):
        streamlines_list = [streamlines_list]

    streamlines_fused, indices = _find_identical_streamlines(
        streamlines_list, epsilon=10**(-precision))
    return streamlines_fused[indices], indices


def difference_robust(streamlines_list, precision=3):
    """ Difference of a list of StatefulTractogram from the first element """
    if not isinstance(streamlines_list, list):
        streamlines_list = [streamlines_list]
    streamlines_fused, indices = _find_identical_streamlines(
        streamlines_list, epsilon=10**(-precision), difference_mode=True)
    return streamlines_fused[indices], indices


def union_robust(streamlines_list, precision=3):
    """ Union of a list of StatefulTractogram """
    if not isinstance(streamlines_list, list):
        streamlines_list = [streamlines_list]
    streamlines_fused, indices = _find_identical_streamlines(
        streamlines_list, epsilon=10**(-precision), union_mode=True)
    return streamlines_fused[indices], indices


def _find_identical_streamlines(streamlines_list, epsilon=0.001,
                                union_mode=False, difference_mode=False):
    """ Return the intersection/union/difference from a list of list of
    streamlines. Allows for a maximum distance for matching.

    Parameters:
    -----------
    streamlines_list: list
        List of lists of streamlines or list of ArraySequences
    epsilon: float
        Maximum allowed distance (should not go above 1.0)
    union_mode: bool
        Perform the union of streamlines
    difference_mode
        Perform the difference of streamlines (from the first element)
    Returns:
    --------
    Tuple, ArraySequence, np.ndarray
        Returns the concatenated streamlines and the indices to pick from it
    """
    streamlines = ArraySequence(itertools.chain(*streamlines_list))
    nb_streamlines = np.cumsum([len(sft) for sft in streamlines_list])
    nb_streamlines = np.insert(nb_streamlines, 0, 0)

    if union_mode and difference_mode:
        raise ValueError('Cannot use union_mode and difference_mode at the '
                         'same time.')
    intersect_mode = (not union_mode and not difference_mode)

    all_tree = {}
    all_tree_mapping = {}
    first_points = np.array(streamlines.get_data()[streamlines._offsets])

    # Uses the number of point to speed up the search in the ckdtree
    for point_count in np.unique(streamlines._lengths):
        same_length_ind = np.where(streamlines._lengths == point_count)[0]
        all_tree[point_count] = cKDTree(first_points[same_length_ind])
        all_tree_mapping[point_count] = same_length_ind

    inversion_val = 1 if union_mode or difference_mode else 0
    streamlines_to_keep = np.ones((len(streamlines),)) * inversion_val
    average_match_distance = []

    # Difference by design will never select streamlines that are not from the
    # first set
    if difference_mode:
        streamlines_to_keep[nb_streamlines[1]:] = 0
    for i, streamline in enumerate(streamlines):
        # Unless we do a union, there is no point looking past the first set
        if not union_mode and i >= nb_streamlines[1]:
            break

        # Find the closest (first) points
        distance_ind = all_tree[len(streamline)].query_ball_point(
            streamline[0], r=2*epsilon)
        actual_ind = np.sort(all_tree_mapping[len(streamline)][distance_ind])

        # Intersection requires finding matches in all sets
        if intersect_mode:
            intersect_test = np.zeros((len(nb_streamlines)-1,))

            # The streamline's set itself is obviously already ok.
            set_i = np.max(np.where(nb_streamlines <= i)[0])
            intersect_test[set_i] = True

        # Looking at similar streamlines only:
        for j in actual_ind:
            # 1) Yourself is never a match.
            #    (For union : always kept)
            #    (For difference: if another is found, we will remove i)
            #    (For intersection: will be kept lower if others are found).
            if i == j:
                continue

            # Actual check of the whole streamline
            sub_vector = streamline-streamlines[j]
            norm = np.linalg.norm(sub_vector, axis=1)
            average_match_distance.append(np.average(sub_vector, axis=0))

            if union_mode:
                # 1) Yourself is not a match
                # 2) If the streamline hasn't been selected (by another match)
                # 3) The streamline is 'identical'
                if streamlines_to_keep[i] == 1 and (norm < 2*epsilon).all():
                    streamlines_to_keep[j] = 0

            elif difference_mode:
                # 1) Yourself is not a match
                # 2) The streamline is 'identical'
                if (norm < 2*epsilon).all():
                    set_j = np.max(np.where(nb_streamlines <= j)[0])

                    # If it is an identical streamline, but from the first set,
                    # it needs to be removed, otherwise remove all instances
                    if set_j == 0:
                        # If it is the first 'encounter', keep it.
                        # Else, remove it.
                        if streamlines_to_keep[actual_ind].all():
                            streamlines_to_keep[j] = 0
                    else:
                        streamlines_to_keep[actual_ind] = 0
            else:  # intersect_mode
                # 1) Yourself is not a match
                # 2) An equivalent streamline has not been selected by another
                #    match.
                # 3) The streamline is 'identical'
                if (not np.any(streamlines_to_keep[actual_ind])
                        and (norm < 2*epsilon).all()):
                    set_j = np.max(np.where(nb_streamlines <= j)[0])
                    # If it is an identical streamline, but from the same set
                    # it needs to be removed (keeping streamlines_to_keep at 0)

                    # Else: will be added only if it is found in all sets
                    if set_i != set_j:
                        intersect_test[set_j] = True

        # Verify that you actually found a match in each set
        if intersect_mode and intersect_test.all():
            # Keeping only the first one; i.
            streamlines_to_keep[i] = 1

    # To facilitate debugging and discovering shifts in data
    if average_match_distance:
        logging.info('Average matches distance: {}mm'.format(
            np.round(np.average(average_match_distance, axis=0), 5)))
    else:
        logging.info('No matches found.')

    return streamlines, np.where(streamlines_to_keep > 0)[0].astype(np.uint32)


def concatenate_sft(sft_list, erase_metadata=False, metadata_fake_init=False):
    """ Concatenate a list of StatefulTractogram together """
    if erase_metadata and metadata_fake_init:
        raise ValueError("You cannot choose both erase_metadata and "
                         "metadata_fake_init")
    if erase_metadata:
        sft_list[0].data_per_point = {}
        sft_list[0].data_per_streamline = {}

    fused_sft = sft_list[0]
    for sft in sft_list[1:]:
        if erase_metadata:
            sft.data_per_point = {}
            sft.data_per_streamline = {}
        elif metadata_fake_init:
            for dps_key in list(sft.data_per_streamline.keys()):
                if dps_key not in sft_list[0].data_per_streamline.keys():
                    del sft.data_per_streamline[dps_key]
            for dpp_key in list(sft.data_per_point.keys()):
                if dpp_key not in sft_list[0].data_per_point.keys():
                    del sft.data_per_point[dpp_key]

            for dps_key in sft_list[0].data_per_streamline.keys():
                if dps_key not in sft.data_per_streamline:
                    arr_shape =\
                        list(sft_list[0].data_per_streamline[dps_key].shape)
                    arr_shape[0] = len(sft)
                    sft.data_per_streamline[dps_key] = np.zeros(arr_shape)
            for dpp_key in sft_list[0].data_per_point.keys():
                if dpp_key not in sft.data_per_point:
                    arr_seq = ArraySequence()
                    arr_seq_shape = list(
                        sft_list[0].data_per_point[dpp_key]._data.shape)
                    arr_seq_shape[0] = len(sft.streamlines._data)
                    arr_seq._data = np.zeros(arr_seq_shape)
                    arr_seq._offsets = sft.streamlines._offsets
                    arr_seq._lengths = sft.streamlines._lengths
                    sft.data_per_point[dpp_key] = arr_seq

        if not metadata_fake_init and \
                not StatefulTractogram.are_compatible(sft, sft_list[0]):
            raise ValueError('Incompatible SFT, check space attributes and '
                             'data_per_point/streamlines.')
        elif not is_header_compatible(sft, sft_list[0]):
            raise ValueError('Incompatible SFT, check space attributes.')

        fused_sft += sft

    return fused_sft


def transform_warp_sft(sft, linear_transfo, target, inverse=False,
                       reverse_op=False, deformation_data=None,
                       remove_invalid=True, cut_invalid=False):
    """ Transform tractogram using an affine Subsequently apply a warp from
    antsRegistration (optional).
    Remove/Cut invalid streamlines to preserve sft validity.

    Parameters
    ----------
    sft: StatefulTractogram
        Stateful tractogram object containing the streamlines to transform.
    linear_transfo: numpy.ndarray
        Linear transformation matrix to apply to the tractogram.
    target: Nifti filepath, image object, header
        Final reference for the tractogram after registration.
    inverse: boolean
        Apply the inverse linear transformation.
    reverse_op: boolean
        Apply both transformation in the reverse order
    deformation_data: np.ndarray
        4D array containing a 3D displacement vector in each voxel.
    remove_invalid: boolean
        Remove the streamlines landing out of the bounding box.
    cut_invalid: boolean
        Cut invalid streamlines rather than removing them. Keep the longest
        segment only.

    Return
    ----------
    new_sft : StatefulTractogram
    """
    # Keep track of the streamlines' original space/origin
    space = sft.space
    origin = sft.origin
    dtype = sft.streamlines._data.dtype

    sft.to_rasmm()
    sft.to_center()

    if len(sft.streamlines) == 0:
        return StatefulTractogram(sft.streamlines, target,
                                  Space.RASMM)

    if inverse:
        linear_transfo = np.linalg.inv(linear_transfo)

    if not reverse_op:
        streamlines = transform_streamlines(sft.streamlines,
                                            linear_transfo)
    else:
        streamlines = sft.streamlines

    if deformation_data is not None:
        if not reverse_op:
            affine, _, _, _ = get_reference_info(target)
        else:
            affine = sft.affine

        # Because of duplication, an iteration over chunks of points is
        # necessary for a big dataset (especially if not compressed)
        streamlines = ArraySequence(streamlines)
        nb_points = len(streamlines._data)
        cur_position = 0
        chunk_size = 1000000
        nb_iteration = int(np.ceil(nb_points/chunk_size))
        inv_affine = np.linalg.inv(affine)

        while nb_iteration > 0:
            max_position = min(cur_position + chunk_size, nb_points)
            points = streamlines._data[cur_position:max_position]

            # To access the deformation information, we need to go in VOX space
            # No need for corner shift since we are doing interpolation
            cur_points_vox = np.array(transform_streamlines(points,
                                                            inv_affine)).T

            x_def = map_coordinates(deformation_data[..., 0],
                                    cur_points_vox.tolist(), order=1)
            y_def = map_coordinates(deformation_data[..., 1],
                                    cur_points_vox.tolist(), order=1)
            z_def = map_coordinates(deformation_data[..., 2],
                                    cur_points_vox.tolist(), order=1)

            # ITK is in LPS and nibabel is in RAS, a flip is necessary for ANTs
            final_points = np.array([-1*x_def, -1*y_def, z_def])
            final_points += np.array(points).T

            streamlines._data[cur_position:max_position] = final_points.T
            cur_position = max_position
            nb_iteration -= 1

    if reverse_op:
        streamlines = transform_streamlines(streamlines, linear_transfo)

    streamlines._data = streamlines._data.astype(dtype)
    new_sft = StatefulTractogram(streamlines, target, Space.RASMM,
                                 data_per_point=sft.data_per_point,
                                 data_per_streamline=sft.data_per_streamline)
    if cut_invalid:
        new_sft, _ = cut_invalid_streamlines(new_sft)
    elif remove_invalid:
        new_sft.remove_invalid_streamlines()

    # Move the streamlines back to the original space/origin
    sft.to_space(space)
    sft.to_origin(origin)

    new_sft.to_space(space)
    new_sft.to_origin(origin)

    return new_sft


def compress_streamlines_wrapper(tractogram, error_rate):
    """
    Compresses the streamlines of a tractogram.
    Supports both nibabel.Tractogram dipy.StatefulTractogram
    or list of streamlines.

    Parameters
    ----------
    tractogram: TrkFile, TckFile, ArraySequence, list
        The tractogram to compress.
    error_rate: float
        The maximum distance (in mm) for point displacement during compression.

    Returns
    -------
    compressed_streamlines: list of np.ndarray
        The compressed streamlines.
    """
    if isinstance(tractogram, (TrkFile, TckFile)):
        return lambda: (compress_streamlines(
            s, error_rate) for s in tractogram.streamlines)
    else:
        if hasattr(tractogram, 'streamlines'):
            tractogram = tractogram.streamlines
        return [compress_streamlines(
            s, error_rate) for s in tractogram]


def upsample_tractogram(sft, nb, point_wise_std=None, tube_radius=None,
                        gaussian=None, error_rate=None, seed=None):
    """
    Generates new streamlines by either adding gaussian noise around
    streamlines' points, or by translating copies of existing streamlines
    by a random amount.

    Parameters
    ----------
    sft : StatefulTractogram
        The tractogram to upsample
    nb : int
        The target number of streamlines in the tractogram.
    point_wise_std : float
        The standard deviation of the gaussian to use to generate point-wise
        noise on the streamlines.
    tube_radius : float
        The radius of the tube used to model the streamlines.
    gaussian: float
        The sigma used for smoothing streamlines.
    error_rate : float
        The maximum distance (in mm) to the original position of any point.
    spline: (float, int)
        Pair of sigma and number of control points used to model each
        streamline as a spline and smooth it.
    seed: int
        Seed for RNG.

    Returns
    -------
    new_sft : StatefulTractogram
        The upsampled tractogram.
    """
    rng = np.random.default_rng(seed)

    # Get the streamlines that will serve as a base for new ones
    resampled_sft = resample_streamlines_step_size(sft, 1)
    new_streamlines = []
    indices = rng.choice(len(resampled_sft), nb, replace=True)
    unique_indices, count = np.unique(indices, return_counts=True)

    # For all selected streamlines, add noise and smooth
    for i, c in zip(unique_indices, count):
        s = resampled_sft.streamlines[i]
        if len(s) < 3:
            new_streamlines.extend(np.repeat(s, c).tolist())
        new_s = parallel_transport_streamline(s, c, tube_radius)

        # Generate smooth noise_factor
        noise = rng.normal(loc=0, scale=point_wise_std,
                           size=len(s))

        # Instead of generating random noise, we fit a polynomial to the
        # noise and use it to generate a spatially smooth noise along the
        # streamline (simply to avoid sharp changes in the noise factor).
        x = np.arange(len(noise))
        poly_coeffs = np.polyfit(x, noise, 3)
        polynomial = Polynomial(poly_coeffs[::-1])
        noise_factor = polynomial(x)

        vec = s - new_s
        vec /= np.linalg.norm(vec, axis=0)
        new_s += vec * np.expand_dims(noise_factor, axis=1)

        if gaussian:
            new_s = [smooth_line_gaussian(s, gaussian) for s in new_s]

        new_streamlines.extend(new_s)

    if error_rate:
        compressed_streamlines = compress_streamlines_wrapper(new_streamlines,
                                                              error_rate)
    else:
        compressed_streamlines = new_streamlines

    new_sft = StatefulTractogram.from_sft(compressed_streamlines, sft)
    return new_sft


def split_sft_sequentially(orig_sft, chunk_sizes):
    """
    Divides a stateful tractogram into n sub-tractograms of sizes defined by
    chunk_sizes. Streamlines are separated sequentially from the initial
    streamlines.

    Parameters
    ----------
    orig_sft: StatefulTractogram
        Initial tractogram to subdivide.
    chunk_sizes: list[int]
        Number of streamlines to keep per chunk.

    Return
    ------
    all_chunks: list[StatefulTractogram]
        The list of sub-tractograms as sfts. The number of tractograms returned
        is len(chunk_sizes).
    """
    if sum(chunk_sizes) > len(orig_sft):
        raise ValueError("You asked for more streamlines than are available.")

    nb_chunks = len(chunk_sizes)

    curr = 0
    sfts = []
    for i in range(nb_chunks):
        nb_str = chunk_sizes[i]
        sfts.append(orig_sft[curr:curr + nb_str])
        curr += chunk_sizes[i]

    return sfts


def split_sft_randomly(orig_sft, chunk_sizes, rng_seed,
                       return_indices_only=False):
    """
    Divides a stateful tractogram into n sub-tractograms of sizes defined by
    chunk_sizes. Streamlines are separated randomly from the initial
    streamlines.

    Parameters
    ----------
    orig_sft: StatefulTractogram
        Initial tractogram to subdivide
    chunk_sizes: int or list[int]
        Number of streamlines to keep (per sub-tractogram if it is a list).
    rng_seed: int
        Random seed.
    return_indices_only: bool
        If true, return a random list of indices. Else, return the Stateful
        Tractogram containing the chosen streamlines.

    Return
    ------
    all_chunks: list[StatefulTractogram] or list[list[int]]
        The list of sub-tractograms as sfts. The number of tractograms returned
        is len(chunk_sizes) + 1, where the last item of the list contains
        streamlines that were not included in any.
        (Or the lists of indices if return_indices_only.)
    """
    if isinstance(chunk_sizes, int):
        chunk_sizes = [chunk_sizes]

    if sum(chunk_sizes) > len(orig_sft):
        raise ValueError("You asked for more streamlines than are available.")

    # Shuffle all streamline indices
    rng = np.random.RandomState(rng_seed)
    ind = np.arange(len(orig_sft.streamlines))
    rng.shuffle(ind)

    # Separate indices.
    final_indices = []
    start = 0
    for next_nb_streamlines in chunk_sizes:
        sub_ind = ind[start:start+next_nb_streamlines]
        final_indices.append(sub_ind)
        start += next_nb_streamlines

    # Append indices not included in any chunk
    final_indices.append(ind[start:])

    if return_indices_only:
        return final_indices

    # Format as sft
    all_sfts = []
    for i in range(len(chunk_sizes) + 1):
        all_sfts.append(orig_sft[final_indices[i]])

    return all_sfts


def split_sft_randomly_per_cluster(orig_sft, chunk_sizes, seed, thresholds):
    """
    Divides a stateful tractogram into n sub-tractograms of sizes defined by
    chunk_sizes. Streamlines are separated randomly from each Quickbundle
    cluster created from the initial streamlines (trying to help
    the randomization to ensure there are streamlines from all bundles in each
    subset).

    Parameters
    ----------
    orig_sft: StatefulTractogram
        Initial tractogram to subdivide
    chunk_sizes: list[int]
        Number of streamlines to keep per chunk. We will ensure that the number
        of streamlines kept per cluster is proportional to the cluster's size.
        Final number will be a good approximation of nb_streamlines, but not
        exact.
    seed: int
        Random seed.
    thresholds: list[float]
        QBx threshold values. Suggestion: [40, 30, 20].

    Returns
    -------
    all_sfts: list[StatefulTractogram]
        The list of sub-tractograms as sfts. The number of tractograms returned
        is len(chunk_sizes) + 1, where the last item of the list contains
        streamlines that were not included in any.
    """

    if sum(chunk_sizes) > len(orig_sft):
        raise ValueError("You asked for more streamlines than are available.")

    # Percent of streamlines to keep per chunk.
    nb_chunks = len(chunk_sizes)
    percent_kept_per_chunk = [nb / len(orig_sft) for nb in chunk_sizes]

    logging.debug("Computing QBx")
    rng = np.random.RandomState(seed)
    clusters = qbx_and_merge(orig_sft.streamlines, thresholds, nb_pts=20,
                             verbose=False, rng=rng)

    logging.info("Done. Now getting list of indices in each of the {} "
                 "cluster.".format(len(clusters)))
    total_indices = [[] for _ in range(nb_chunks + 1)]
    for cluster in clusters:
        if len(cluster.indices) > 1:
            cluster_sft = orig_sft[cluster.indices]
            size_cluster = len(cluster.indices)
            chunk_sizes_in_cluster = \
                [round(p * size_cluster) for p in percent_kept_per_chunk]

            # If rounding created too many streamlines, removing some from the
            # last chunk.
            while sum(chunk_sizes_in_cluster) > size_cluster:
                chunk_sizes_in_cluster[-1] -= 1

            all_chunks_inds_in_cluster = split_sft_randomly(
                cluster_sft, chunk_sizes_in_cluster, seed,
                return_indices_only=True)

            assert len(all_chunks_inds_in_cluster) == nb_chunks + 1

            for i in range(nb_chunks + 1):
                chunk_orig_inds = [cluster.indices[ind] for ind in
                                   all_chunks_inds_in_cluster[i]]
                total_indices[i].extend(chunk_orig_inds)

    final_sfts = [orig_sft[inds] for inds in total_indices]

    return final_sfts


OPERATIONS = {
    'difference_robust': difference_robust,
    'intersection_robust': intersection_robust,
    'union_robust': union_robust,
    'difference': difference,
    'intersection': intersection,
    'union': union,
    'concatenate': 'concatenate',
    'lazy_concatenate': 'lazy_concatenate'
}
