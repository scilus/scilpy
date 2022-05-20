# -*- coding: utf-8 -*-
import copy
import itertools
from functools import reduce
import logging


from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import get_reference_info, is_header_compatible
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamlinespeed import compress_streamlines
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from scilpy.tracking.tools import smooth_line_gaussian, smooth_line_spline
from scilpy.tractanalysis.features import get_streamlines_centroid
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

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
    result = right.copy()
    result.update(left)
    return result


def uniformize_bundle_sft(sft, axis=None, swap=False):
    """Uniformize the streamlines in the given tractogram.

    Parameters
    ----------
    sft: StatefulTractogram
         The tractogram that contains the list of streamlines to be uniformized
    axis: int, optional
        Orient endpoints in the given axis
    swap: boolean, optional
        Swap the orientation of streamlines

    """
    if len(sft.streamlines) > 0:
        axis_name = ['x', 'y', 'z']
        if axis is None:
            centroid = get_streamlines_centroid(sft.streamlines, 20)[0]
            main_dir_ends = np.argmax(np.abs(centroid[0] - centroid[-1]))
            main_dir_displacement = np.argmax(
                np.abs(np.sum(np.gradient(centroid, axis=0), axis=0)))
            if main_dir_displacement != main_dir_ends:
                logging.info('Ambiguity in orientation, you should use --axis')
            axis = axis_name[main_dir_displacement]
        logging.info('Orienting endpoints in the {} axis'.format(axis))
        axis_pos = axis_name.index(axis)
        for i in range(len(sft.streamlines)):
            # Bitwise XOR
            if bool(sft.streamlines[i][0][axis_pos] >
                    sft.streamlines[i][-1][axis_pos]) ^ bool(swap):
                sft.streamlines[i] = sft.streamlines[i][::-1]
                for key in sft.data_per_point[i]:
                    sft.data_per_point[key][i] = \
                        sft.data_per_point[key][i][::-1]


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
    indices: np.ndarray
        The indices of the streamlines that are used in the output.

    """

    # Hash the streamlines using the desired precision.
    indices = np.cumsum([0] + [len(s) for s in streamlines[:-1]])
    hashes = [hash_streamlines(s, i, precision) for
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

    streamlines_fused, indices = find_identical_streamlines(streamlines_list,
                                                            epsilon=10**(-precision))
    return streamlines_fused[indices], indices


def difference_robust(streamlines_list, precision=3):
    """ Difference of a list of StatefulTractogram from the first element """
    if not isinstance(streamlines_list, list):
        streamlines_list = [streamlines_list]
    streamlines_fused, indices = find_identical_streamlines(streamlines_list,
                                                            epsilon=10**(-precision),
                                                            difference_mode=True)
    return streamlines_fused[indices], indices


def union_robust(streamlines_list, precision=3):
    """ Union of a list of StatefulTractogram """
    if not isinstance(streamlines_list, list):
        streamlines_list = [streamlines_list]
    streamlines_fused, indices = find_identical_streamlines(streamlines_list,
                                                            epsilon=10**(-precision),
                                                            union_mode=True)
    return streamlines_fused[indices], indices


def find_identical_streamlines(streamlines_list, epsilon=0.001,
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
        # Unless do an union, there is no point at looking past the first set
        if not union_mode and i >= nb_streamlines[1]:
            break

        # Find the closest (first) points
        distance_ind = all_tree[len(streamline)].query_ball_point(streamline[0],
                                                                  r=2*epsilon)
        actual_ind = np.sort(all_tree_mapping[len(streamline)][distance_ind])

        # Intersection requires finding matches is all sets
        if not union_mode or not difference_mode:
            intersect_test = np.zeros((len(nb_streamlines)-1,))

        for j in actual_ind:
            # Actual check of the whole streamline
            sub_vector = streamline-streamlines[j]
            norm = np.linalg.norm(sub_vector, axis=1)

            if union_mode:
                # 1) Yourself is not a match
                # 2) If the streamline hasn't been selected (by another match)
                # 3) The streamline is 'identical'
                if i != j and streamlines_to_keep[i] == inversion_val \
                        and (norm < 2*epsilon).all():
                    streamlines_to_keep[j] = not inversion_val
                    average_match_distance.append(np.average(sub_vector,
                                                             axis=0))
            elif difference_mode:
                # 1) Yourself is not a match
                # 2) The streamline is 'identical'
                if i != j and (norm < 2*epsilon).all():
                    pos_in_list_j = np.max(np.where(nb_streamlines <= j)[0])

                    # If it is an identical streamline, but from the same set
                    # it needs to be removed, otherwise remove all instances
                    if pos_in_list_j == 0:
                        # If it is the first 'encounter' add it
                        if streamlines_to_keep[actual_ind].all():
                            streamlines_to_keep[j] = not inversion_val
                            average_match_distance.append(np.average(sub_vector,
                                                                     axis=0))
                    else:
                        streamlines_to_keep[actual_ind] = not inversion_val
                        average_match_distance.append(np.average(sub_vector,
                                                                 axis=0))
            else:
                # 1) The streamline is 'identical'
                if (norm < 2*epsilon).all():
                    pos_in_list_i = np.max(np.where(nb_streamlines <= i)[0])
                    pos_in_list_j = np.max(np.where(nb_streamlines <= j)[0])
                    # If it is an identical streamline, but from the same set
                    # it needs to be removed
                    if i == j or pos_in_list_i != pos_in_list_j:
                        intersect_test[pos_in_list_j] = True
                    if i != j:
                        average_match_distance.append(
                            np.average(sub_vector, axis=0))

        # Verify that you actually found a match in each set
        if (not union_mode or not difference_mode) and intersect_test.all():
            streamlines_to_keep[i] = not inversion_val

    # To facilitate debugging and discovering shifts in data
    if average_match_distance:
        logging.info('Average matches distance: {}mm'.format(
            np.round(np.average(average_match_distance, axis=0), 5)))
    else:
        logging.info('No matches found.')

    return streamlines, np.where(streamlines_to_keep > 0)[0].astype(np.uint32)


def concatenate_sft(sft_list, erase_metadata=False, metadata_fake_init=False):
    """ Concatenate a list of StatefulTractogram together """
    if erase_metadata:
        sft_list[0].data_per_point = {}
        sft_list[0].data_per_streamline = {}

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

    total_streamlines = 0
    total_points = 0
    lengths = []
    for sft in sft_list:
        total_streamlines += len(sft.streamlines._offsets)
        total_points += len(sft.streamlines._data)
        lengths.extend(sft.streamlines._lengths)
    lengths = np.array(lengths, dtype=np.uint32)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.uint64)

    dpp = {}
    for dpp_key in sft_list[0].data_per_point.keys():
        arr_seq_shape = list(sft_list[0].data_per_point[dpp_key]._data.shape)
        arr_seq_shape[0] = total_points
        dpp[dpp_key] = ArraySequence()
        dpp[dpp_key]._data = np.zeros(arr_seq_shape)
        dpp[dpp_key]._lengths = lengths
        dpp[dpp_key]._offsets = offsets

    dps = {}
    for dps_key in sft_list[0].data_per_streamline.keys():
        arr_seq_shape = list(sft_list[0].data_per_streamline[dps_key].shape)
        arr_seq_shape[0] = total_streamlines
        dps[dps_key] = np.zeros(arr_seq_shape)

    streamlines = ArraySequence()
    streamlines._data = np.zeros((total_points, 3))
    streamlines._lengths = lengths
    streamlines._offsets = offsets

    pts_counter = 0
    strs_counter = 0
    for sft in sft_list:
        pts_curr_len = len(sft.streamlines._data)
        strs_curr_len = len(sft.streamlines._offsets)

        if strs_curr_len == 0 or pts_curr_len == 0:
            continue

        streamlines._data[pts_counter:pts_counter+pts_curr_len] = \
            sft.streamlines._data

        for dpp_key in sft_list[0].data_per_point.keys():
            dpp[dpp_key]._data[pts_counter:pts_counter+pts_curr_len] = \
                sft.data_per_point[dpp_key]._data
        for dps_key in sft_list[0].data_per_streamline.keys():
            dps[dps_key][strs_counter:strs_counter+strs_curr_len] = \
                sft.data_per_streamline[dps_key]
        pts_counter += pts_curr_len
        strs_counter += strs_curr_len

    fused_sft = StatefulTractogram.from_sft(streamlines, sft_list[0],
                                            data_per_point=dpp,
                                            data_per_streamline=dps)
    return fused_sft


def transform_warp_sft(sft, linear_transfo, target, inverse=False,
                       reverse_op=False, deformation_data=None,
                       remove_invalid=True, cut_invalid=False):
    """ Transform tractogram using a affine Subsequently apply a warp from
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
        streamlines = transform_streamlines(streamlines,
                                            linear_transfo)

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

    streamline_ids = np.asarray(streamline_ids, dtype=int)

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
    This is a tradeoff to speed up the linearization process [Rheault15]_. A
    low value will result in a faster linearization but low compression,
    whereas a high value will result in a slower linearization but high
    compression.

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
        logging.warning("Initial StatefulTractogram contained data_per_point. "
                        "This information will not be carried in the final"
                        "tractogram.")

    compressed_sft = StatefulTractogram.from_sft(
        compressed_streamlines, sft,
        data_per_streamline=sft.data_per_streamline)

    # Return to original space
    compressed_sft.to_space(orig_space)

    return compressed_sft


def cut_invalid_streamlines(sft):
    """ Cut streamlines so their longest segment are within the bounding box.
    This function keeps the data_per_point and data_per_streamline.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to remove invalid points from.

    Returns
    -------
    new_sft : StatefulTractogram
        New object with the invalid points removed from each streamline.
    cutting_counter : int
        Number of streamlines that were cut.
    """
    if not len(sft):
        return sft, 0

    # Keep track of the streamlines' original space/origin
    space = sft.space
    origin = sft.origin

    sft.to_vox()
    sft.to_corner()

    copy_sft = copy.deepcopy(sft)
    epsilon = 0.001
    indices_to_remove, _ = copy_sft.remove_invalid_streamlines()

    new_streamlines = []
    new_data_per_point = {}
    new_data_per_streamline = {}
    for key in sft.data_per_point.keys():
        new_data_per_point[key] = []
    for key in sft.data_per_streamline.keys():
        new_data_per_streamline[key] = []

    cutting_counter = 0
    for ind in range(len(sft.streamlines)):
        # No reason to try to cut if all points are within the volume
        if ind in indices_to_remove:
            best_pos = [0, 0]
            cur_pos = [0, 0]
            for pos, point in enumerate(sft.streamlines[ind]):
                if (point < epsilon).any() or \
                        (point >= sft.dimensions - epsilon).any():
                    cur_pos = [pos+1, pos+1]
                if cur_pos[1] - cur_pos[0] > best_pos[1] - best_pos[0]:
                    best_pos = cur_pos
                cur_pos[1] += 1

            if not best_pos == [0, 0]:
                new_streamlines.append(
                    sft.streamlines[ind][best_pos[0]:best_pos[1]-1])
                cutting_counter += 1
                for key in sft.data_per_streamline.keys():
                    new_data_per_streamline[key].append(
                        sft.data_per_streamline[key][ind])
                for key in sft.data_per_point.keys():
                    new_data_per_point[key].append(
                        sft.data_per_point[key][ind][best_pos[0]:best_pos[1]-1])
            else:
                logging.warning('Streamlines entirely out of the volume.')
        else:
            new_streamlines.append(sft.streamlines[ind])
            for key in sft.data_per_streamline.keys():
                new_data_per_streamline[key].append(
                    sft.data_per_streamline[key][ind])
            for key in sft.data_per_point.keys():
                new_data_per_point[key].append(sft.data_per_point[key][ind])
    new_sft = StatefulTractogram.from_sft(new_streamlines, sft,
                                          data_per_streamline=new_data_per_streamline,
                                          data_per_point=new_data_per_point)

    # Move the streamlines back to the original space/origin
    sft.to_space(space)
    sft.to_origin(origin)

    new_sft.to_space(space)
    new_sft.to_origin(origin)

    return new_sft, cutting_counter


def upsample_tractogram(
    sft, nb, point_wise_std=None,
    streamline_wise_std=None, gaussian=None, spline=None, seed=None
):
    """
    Generate new streamlines by either adding gaussian noise around
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
    streamline_wise_std : float
        The standard deviation of the gaussian to use to generate
        streamline-wise noise on the streamlines.
    gaussian: float
        The sigma used for smoothing streamlines.
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
    assert bool(point_wise_std) ^ bool(streamline_wise_std), \
        'Can only add either point-wise or streamline-wise noise' + \
        ', not both nor none.'

    rng = np.random.RandomState(seed)

    # Get the number of streamlines to add
    nb_new = nb - len(sft.streamlines)

    # Get the streamlines that will serve as a base for new ones
    indices = rng.choice(
        len(sft.streamlines), nb_new)
    new_streamlines = sft.streamlines.copy()

    # For all selected streamlines, add noise and smooth
    for s in sft.streamlines[indices]:
        if point_wise_std:
            noise = rng.normal(scale=point_wise_std, size=s.shape)
        elif streamline_wise_std:
            noise = rng.normal(
                scale=streamline_wise_std, size=s.shape[-1])
        new_s = s + noise
        if gaussian:
            new_s = smooth_line_gaussian(new_s, gaussian)
        elif spline:
            new_s = smooth_line_spline(new_s, spline[0],
                                       spline[1])

        new_streamlines.append(new_s)

    new_sft = StatefulTractogram.from_sft(new_streamlines, sft)
    return new_sft
