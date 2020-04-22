# -*- coding: utf-8 -*-

import itertools
import logging

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamlinespeed import compress_streamlines
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def sum_sft(sft_list, erase_metadata=False):
    """ Concatenate a list of StatefulTractogram together """
    fused_sft = sft_list[0]
    if erase_metadata:
        fused_sft.data_per_point = {}
        fused_sft.data_per_streamline = {}

    for sft in sft_list[1:]:
        if erase_metadata:
            sft.data_per_point = {}
            sft.data_per_streamline = {}

        if not StatefulTractogram.is_compatible_sft(sft, fused_sft):
            raise ValueError('Incompatible SFT.')
        fused_sft += sft

    return fused_sft


def intersection(streamlines_list, precision=3):
    """ Intersection of a list of StatefulTractogram """
    if not isinstance(streamlines_list, list):
        streamlines_list = [streamlines_list]

    streamlines_fused, indices = find_identical_streamlines(streamlines_list,
                                                            epsilon=10**(-precision))
    return streamlines_fused[indices], indices


def difference(streamlines_list, precision=3):
    """ Difference of a list of StatefulTractogram from the first element """
    if not isinstance(streamlines_list, list):
        streamlines_list = [streamlines_list]
    streamlines_fused, indices = find_identical_streamlines(streamlines_list,
                                                            epsilon=10**(-precision),
                                                            difference_mode=True)
    return streamlines_fused[indices], indices


def union(streamlines_list, precision=3):
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

    return streamlines, np.where(streamlines_to_keep > 0)[0]


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
