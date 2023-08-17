# -*- coding: utf-8 -*-
import logging

import numpy as np
import scipy.ndimage as ndi
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import (length, set_number_of_points)
from scipy.interpolate import splev, splprep
from sklearn.cluster import KMeans

from scilpy.tractanalysis.reproducibility_measures import get_endpoints_density_map
from scilpy.tractanalysis.uncompress import uncompress
from scilpy.tractanalysis.quick_tools import (get_next_real_point,
                                              get_previous_real_point)


def get_streamline_pt_index(points_to_index, vox_index, from_start=True):
    cur_idx = np.where(points_to_index == vox_index)

    if not len(cur_idx[0]):
        return None

    if from_start:
        idx_to_take = 0
    else:
        idx_to_take = -1

    return cur_idx[0][idx_to_take]


def get_point_on_line(first_point, second_point, vox_lower_corner):
    # To manage the case where there is no real streamline point in an
    # intersected voxel, we need to generate an artificial point.
    # We use line / cube intersections as presented in
    # Physically Based Rendering, Second edition, pp. 192-195
    # Some simplifications are made since we are sure that an intersection
    # exists (else this function would not have been called).
    ray = second_point - first_point
    ray /= np.linalg.norm(ray)

    corners = np.array([vox_lower_corner, vox_lower_corner + 1])

    t0 = 0
    t1 = np.inf
    for i in range(3):
        if ray[i] != 0.:
            inv_ray = 1. / ray[i]
            v0 = (corners[0, i] - first_point[i]) * inv_ray
            v1 = (corners[1, i] - first_point[i]) * inv_ray
            t0 = max(t0, min(v0, v1))
            t1 = min(t1, max(v0, v1))

    return first_point + ray * (t0 + t1) / 2.

def filter_streamlines_by_length(sft, min_length=0., max_length=np.inf):
    """
    Filter streamlines using minimum and max length.

    Parameters
    ----------
    sft: StatefulTractogram
        SFT containing the streamlines to filter.
    min_length: float
        Minimum length of streamlines, in mm.
    max_length: float
        Maximum length of streamlines, in mm.

    Return
    ------
    filtered_sft : StatefulTractogram
        A tractogram without short streamlines.
    """

    # Make sure we are in world space
    orig_space = sft.space
    sft.to_rasmm()

    if sft.streamlines:
        # Compute streamlines lengths
        lengths = length(sft.streamlines)

        # Filter lengths
        filter_stream = np.logical_and(lengths >= min_length,
                                       lengths <= max_length)
    else:
        filter_stream = []

    filtered_sft = sft[filter_stream]

    # Return to original space
    filtered_sft.to_space(orig_space)

    return filtered_sft


def filter_streamlines_by_total_length_per_dim(
        sft, limits_x, limits_y, limits_z, use_abs, save_rejected):
    """
    Filter streamlines using sum of abs length per dimension.

    Note: we consider that x, y, z are the coordinates of the streamlines; we
    do not verify if they are aligned with the brain's orientation.

    Parameters
    ----------
    sft: StatefulTractogram
        SFT containing the streamlines to filter.
    limits_x: [float float]
        The list of [min, max] for the x coordinates.
    limits_y: [float float]
        The list of [min, max] for the y coordinates.
    limits_z: [float float]
        The list of [min, max] for the z coordinates.
    use_abs: bool
        If True, will use the total of distances in absolute value (ex,
        coming back on yourself will contribute to the total distance
        instead of cancelling it).
    save_rejected: bool
        If true, also returns the SFT of rejected streamlines. Else, returns
        None.

    Return
    ------
    filtered_sft : StatefulTractogram
        A tractogram of accepted streamlines.
    ids: list
        The list of good ids.
    rejected_sft: StatefulTractogram or None
        A tractogram of rejected streamlines.
    """
    # Make sure we are in world space
    orig_space = sft.space
    sft.to_rasmm()

    # Compute directions
    all_dirs = [np.diff(s, axis=0) for s in sft.streamlines]
    if use_abs:
        total_per_orientation = np.asarray(
            [np.sum(np.abs(d), axis=0) for d in all_dirs])
    else:
        # We add the abs on the total length, not on each small movement.
        total_per_orientation = np.abs(np.asarray(
            [np.sum(d, axis=0) for d in all_dirs]))

    logging.debug("Total length per orientation is:\n"
                  "Average: x: {:.2f}, y: {:.2f}, z: {:.2f} \n"
                  "Min: x: {:.2f}, y: {:.2f}, z: {:.2f} \n"
                  "Max: x: {:.2f}, y: {:.2f}, z: {:.2f} \n"
                  .format(np.mean(total_per_orientation[:, 0]),
                          np.mean(total_per_orientation[:, 1]),
                          np.mean(total_per_orientation[:, 2]),
                          np.min(total_per_orientation[:, 0]),
                          np.min(total_per_orientation[:, 1]),
                          np.min(total_per_orientation[:, 2]),
                          np.max(total_per_orientation[:, 0]),
                          np.max(total_per_orientation[:, 1]),
                          np.max(total_per_orientation[:, 2])))

    # Find good ids
    mask_good_x = np.logical_and(limits_x[0] < total_per_orientation[:, 0],
                                 total_per_orientation[:, 0] < limits_x[1])
    mask_good_y = np.logical_and(limits_y[0] < total_per_orientation[:, 1],
                                 total_per_orientation[:, 1] < limits_y[1])
    mask_good_z = np.logical_and(limits_z[0] < total_per_orientation[:, 2],
                                 total_per_orientation[:, 2] < limits_z[1])
    mask_good_ids = np.logical_and(mask_good_x, mask_good_y)
    mask_good_ids = np.logical_and(mask_good_ids, mask_good_z)

    filtered_sft = sft[mask_good_ids]

    rejected_sft = None
    if save_rejected:
        rejected_sft = sft[~mask_good_ids]

    # Return to original space
    filtered_sft.to_space(orig_space)

    return filtered_sft, np.nonzero(mask_good_ids), rejected_sft


def resample_streamlines_num_points(sft, num_points):
    """
    Resample streamlines using number of points per streamline

    Parameters
    ----------
    sft: StatefulTractogram
        SFT containing the streamlines to subsample.
    num_points: int
        Number of points per streamline in the output.

    Return
    ------
    resampled_sft: StatefulTractogram
        The resampled streamlines as a sft.
    """

    # Checks
    if num_points <= 1:
        raise ValueError("The value of num_points should be greater than 1!")

    # Resampling
    resampled_streamlines = []
    for streamline in sft.streamlines:
        line = set_number_of_points(streamline, num_points)
        resampled_streamlines.append(line)

    # Creating sft
    # CAREFUL. Data_per_point will be lost.
    resampled_sft = _warn_and_save(resampled_streamlines, sft)

    return resampled_sft


def resample_streamlines_step_size(sft, step_size):
    """
    Resample streamlines using a fixed step size.

    Parameters
    ----------
    sft: StatefulTractogram
        SFT containing the streamlines to subsample.
    step_size: float
        Size of the new steps, in mm.

    Return
    ------
    resampled_sft: StatefulTractogram
        The resampled streamlines as a sft.
    """

    # Checks
    if step_size == 0:
        raise ValueError("Step size can't be 0!")
    elif step_size < 0.1:
        logging.debug("The value of your step size seems suspiciously low. "
                      "Please check.")
    elif step_size > np.max(sft.voxel_sizes):
        logging.debug("The value of your step size seems suspiciously high. "
                      "Please check.")

    # Make sure we are in world space
    orig_space = sft.space
    sft.to_rasmm()

    # Resampling
    lengths = length(sft.streamlines)
    nb_points = np.ceil(lengths / step_size).astype(int)
    if np.any(nb_points == 1):
        logging.warning("Some streamlines are shorter than the provided "
                        "step size...")
        nb_points[nb_points == 1] = 2
    resampled_streamlines = [set_number_of_points(s, n) for s, n in
                             zip(sft.streamlines, nb_points)]

    # Creating sft
    resampled_sft = _warn_and_save(resampled_streamlines, sft)

    # Return to original space
    resampled_sft.to_space(orig_space)

    return resampled_sft


def _warn_and_save(new_streamlines, sft):
    """Last step of the two resample functions:
    Warn that we loose data_per_point, then create resampled SFT."""

    if sft.data_per_point is not None and sft.data_per_point.keys():
        logging.debug("Initial StatefulTractogram contained data_per_point. "
                      "This information will not be carried in the final "
                      "tractogram.")
    new_sft = StatefulTractogram.from_sft(
        new_streamlines, sft, data_per_streamline=sft.data_per_streamline)

    return new_sft


def cut_outside_of_mask_streamlines(sft, binary_mask, min_len=0):
    """ Cut streamlines so their longest segment are within the bounding box
    or a binary mask.
    This function erases the data_per_point and data_per_streamline.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to cut streamlines (using a single mask with 1 entity) from.
    binary_mask: np.ndarray
        Boolean array representing the region (must contain 1 entity)
    min_len: float
        Minimum length from the resulting streamlines.

    Returns
    -------
    new_sft : StatefulTractogram
        New object with the streamlines trimmed within the mask.
    """
    sft.to_vox()
    sft.to_corner()
    streamlines = sft.streamlines

    new_streamlines = []
    for _, streamline in enumerate(streamlines):
        entry_found = False
        last_success = 0
        curr_len = 0
        longest_seq = (0, 0)
        for ind, pos in enumerate(streamline):
            pos = tuple(pos.astype(np.int16))
            if binary_mask[pos]:
                if not entry_found:
                    entry_found = True
                    last_success = ind
                    curr_len = 0
                else:
                    curr_len += 1
                    if curr_len > longest_seq[1] - longest_seq[0]:
                        longest_seq = (last_success, ind + 1)
            else:
                if entry_found:
                    entry_found = False
                    if curr_len > longest_seq[1] - longest_seq[0]:
                        longest_seq = (last_success, ind)
                        curr_len = 0
        # print(longest_seq)
        if longest_seq[1] != 0:
            new_streamlines.append(streamline[longest_seq[0]:longest_seq[1]])

    new_sft = StatefulTractogram.from_sft(new_streamlines, sft)
    return filter_streamlines_by_length(new_sft, min_length=min_len)


def cut_between_masks_streamlines(sft, binary_mask, min_len=0):
    """ Cut streamlines so their segment are within the bounding box
    or going from binary mask #1 to binary mask #2.
    This function erases the data_per_point and data_per_streamline.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to cut streamlines (using a single mask with 2 entities) from.
    binary_mask: np.ndarray
        Boolean array representing the region (must contain 2 entities)
    min_len: float
        Minimum length from the resulting streamlines.
    Returns
    -------
    new_sft : StatefulTractogram
        New object with the streamlines trimmed within the masks.
    """
    sft.to_vox()
    sft.to_corner()
    streamlines = sft.streamlines

    density = get_endpoints_density_map(streamlines, binary_mask.shape)
    density[density > 0] = 1
    density[binary_mask == 0] = 0

    roi_data_1, roi_data_2 = split_heads_tails_kmeans(binary_mask)

    new_streamlines = []
    (indices, points_to_idx) = uncompress(streamlines, return_mapping=True)

    for strl_idx, strl in enumerate(streamlines):
        strl_indices = indices[strl_idx]

        in_strl_idx, out_strl_idx = intersects_two_rois(roi_data_1,
                                                        roi_data_2,
                                                        strl_indices)

        if in_strl_idx is not None and out_strl_idx is not None:
            points_to_indices = points_to_idx[strl_idx]
            tmp = compute_streamline_segment(strl, strl_indices, in_strl_idx,
                                             out_strl_idx, points_to_indices)
            new_streamlines.append(tmp)

    new_sft = StatefulTractogram.from_sft(new_streamlines, sft)
    return filter_streamlines_by_length(new_sft, min_length=min_len)


def compute_streamline_segment(orig_strl, inter_vox, in_vox_idx, out_vox_idx,
                               points_to_indices):
    additional_start_pt = None
    additional_exit_pt = None
    nb_points = 0

    # Check if the indexed voxel contains a real streamline point
    in_strl_point = get_streamline_pt_index(points_to_indices,
                                            in_vox_idx)

    if in_strl_point is None:
        # Find the next real streamline point
        in_strl_point = get_next_real_point(points_to_indices, in_vox_idx)

        additional_start_pt = get_point_on_line(orig_strl[in_strl_point - 1],
                                                orig_strl[in_strl_point],
                                                inter_vox[in_vox_idx])
        nb_points += 1

    # Generate point for the current voxel
    exit_strl_point = get_streamline_pt_index(points_to_indices,
                                              out_vox_idx,
                                              from_start=False)

    if exit_strl_point is None:
        # Find the previous real streamline point
        exit_strl_point = get_previous_real_point(points_to_indices,
                                                  out_vox_idx)

        additional_exit_pt = get_point_on_line(orig_strl[exit_strl_point],
                                               orig_strl[exit_strl_point + 1],
                                               inter_vox[out_vox_idx])
        nb_points += 1

    if exit_strl_point >= in_strl_point:
        nb_points_orig_strl = exit_strl_point - in_strl_point + 1
        nb_points += nb_points_orig_strl

    segment = np.zeros((nb_points, 3))
    at_point = 0

    if additional_start_pt is not None:
        segment[0] = additional_start_pt
        at_point += 1

    if exit_strl_point >= in_strl_point:
        # Note: this works correctly even in the case where the "previous"
        # point is the same or lower than the entry point, because of
        # numpy indexing
        segment[at_point:at_point + nb_points_orig_strl] = \
            orig_strl[in_strl_point:exit_strl_point + 1]
        at_point += nb_points_orig_strl

    if additional_exit_pt is not None:
        segment[at_point] = additional_exit_pt
        at_point += 1

    return segment


def intersects_two_rois(roi_data_1, roi_data_2, strl_indices):
    """ Cut streamlines so their longest segment are within the bounding box
    or a binary mask.
    This function keeps the data_per_point and data_per_streamline.

    Parameters
    ----------
    roi_data_1: np.ndarray
        Boolean array representing the region #1
    roi_data_2: np.ndarray
        Boolean array representing the region #2
    strl_indices: list of tuple (3xint)
        3D indices of the voxel intersected by the streamline

    Returns
    -------
    in_strl_idx : int
        index of the first point (of the streamline) to be in the masks
    out_strl_idx : int
        index of the last point (of the streamline) to be in the masks
    """
    entry_found = False
    exit_found = False
    went_out_of_exit = False
    exit_roi_data = None
    in_strl_idx = None
    out_strl_idx = None

    for idx, point in enumerate(strl_indices):
        if entry_found and exit_found:
            # Still add points that are in exit roi, to mimic entry ROI
            # This will need to be modified to correctly handle continuation
            if exit_roi_data[tuple(point)] > 0:
                if not went_out_of_exit:
                    out_strl_idx = idx
            else:
                went_out_of_exit = True
        elif entry_found and not exit_found:
            # If we reached the exit ROI
            if exit_roi_data[tuple(point)] > 0:
                exit_found = True
                out_strl_idx = idx
        elif not entry_found:
            # Check if we are in one of ROIs
            if roi_data_1[tuple(point)] > 0 or roi_data_2[tuple(point)] > 0:
                entry_found = True
                in_strl_idx = idx
                if roi_data_1[tuple(point)] > 0:
                    exit_roi_data = roi_data_2
                else:
                    exit_roi_data = roi_data_1

    return in_strl_idx, out_strl_idx


def split_heads_tails_kmeans(data):
    """
    Split a mask between head and tail with k means.

    Parameters
    ----------
    data: numpy.ndarray
        Mask to be split.

    Returns
    -------
    mask_1: numpy.ndarray
        "Head" of the mask.
    mask_2: numpy.ndarray
        "Tail" of the mask.
    """

    X = np.argwhere(data)
    k_means = KMeans(n_clusters=2).fit(X)
    mask_1 = np.zeros(data.shape)
    mask_2 = np.zeros(data.shape)

    mask_1[tuple(X[np.where(k_means.labels_ == 0)].T)] = 1
    mask_2[tuple(X[np.where(k_means.labels_ == 1)].T)] = 1

    return mask_1, mask_2


def smooth_line_gaussian(streamline, sigma):
    if sigma < 0.00001:
        ValueError('Cant have a 0 sigma with gaussian.')

    nb_points = int(length(streamline))
    if nb_points < 2:
        logging.debug('Streamline shorter than 1mm, corner cases possible.')
        nb_points = 2
    sampled_streamline = set_number_of_points(streamline, nb_points)

    x, y, z = sampled_streamline.T
    x3 = ndi.gaussian_filter1d(x, sigma)
    y3 = ndi.gaussian_filter1d(y, sigma)
    z3 = ndi.gaussian_filter1d(z, sigma)
    smoothed_streamline = np.asarray([x3, y3, z3], dtype=float).T

    # Ensure first and last point remain the same
    smoothed_streamline[0] = streamline[0]
    smoothed_streamline[-1] = streamline[-1]

    return smoothed_streamline


def smooth_line_spline(streamline, sigma, nb_ctrl_points):
    if sigma < 0.00001:
        ValueError('Cant have a 0 sigma with spline.')

    nb_points = int(length(streamline))
    if nb_points < 2:
        logging.debug('Streamline shorter than 1mm, corner cases possible.')

    if nb_ctrl_points < 3:
        nb_ctrl_points = 3

    sampled_streamline = set_number_of_points(streamline, nb_ctrl_points)

    tck, u = splprep(sampled_streamline.T, s=sigma)
    smoothed_streamline = splev(np.linspace(0, 1, 99), tck)
    smoothed_streamline = np.squeeze(np.asarray([smoothed_streamline]).T)

    # Ensure first and last point remain the same
    smoothed_streamline[0] = streamline[0]
    smoothed_streamline[-1] = streamline[-1]

    return smoothed_streamline
