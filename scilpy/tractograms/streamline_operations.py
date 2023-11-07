# -*- coding: utf-8 -*-
import logging

import numpy as np
import scipy.ndimage as ndi
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import (length, set_number_of_points)
from scipy.interpolate import splev, splprep


def _get_streamline_pt_index(points_to_index, vox_index, from_start=True):
    """ Get the index of the streamline point in the voxel.

    Arguments
    ---------
    points_to_index: np.ndarray
        The indices of the voxels in the streamline's voxel grid.
    vox_index: int
        The index of the voxel in the voxel grid.
    from_start: bool
        If True, will return the first streamline point in the voxel.
        If False, will return the last streamline point in the voxel.

    Returns
    -------
    index: int or None
        The index of the streamline point in the voxel.
        If None, there is no streamline point in the voxel.
    """

    cur_idx = np.where(points_to_index == vox_index)

    if not len(cur_idx[0]):
        return None

    if from_start:
        idx_to_take = 0
    else:
        idx_to_take = -1

    return cur_idx[0][idx_to_take]


def _get_point_on_line(first_point, second_point, vox_lower_corner):
    """ Get the point on a line that is in a voxel.

    To manage the case where there is no real streamline point in an
    intersected voxel, we need to generate an artificial point.
    We use line / cube intersections as presented in
    Physically Based Rendering, Second edition, pp. 192-195
    Some simplifications are made since we are sure that an intersection
    exists (else this function would not have been called).

    Arguments
    ---------
    first_point: np.ndarray
        The first point of the line.
    second_point: np.ndarray
        The second point of the line.
    vox_lower_corner: np.ndarray
        The lower corner coordinates of the voxel.

    Returns
    -------
    intersection_point: np.ndarray
        The point on the line that is in the voxel.
    """

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
    sft.to_space(orig_space)
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
    lines = set_number_of_points(sft.streamlines, num_points)

    # Creating sft
    # CAREFUL. Data_per_point will be lost.
    resampled_sft = _warn_and_save(lines, sft)

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


def smooth_line_gaussian(streamline, sigma):
    """ Smooth a streamline using a gaussian filter.

    Parameters
    ----------
    streamline: np.ndarray
        The streamline to smooth.
    sigma: float
        The sigma of the gaussian filter.

    Returns
    -------
    smoothed_streamline: np.ndarray
        The smoothed streamline.
    """

    if sigma < 0.00001:
        raise ValueError('Cant have a 0 sigma with gaussian.')

    if length(streamline) < 1:
        logging.debug('Streamline shorter than 1mm, corner cases possible.')

    # Smooth each dimension separately
    x, y, z = streamline.T
    x3 = ndi.gaussian_filter1d(x, sigma)
    y3 = ndi.gaussian_filter1d(y, sigma)
    z3 = ndi.gaussian_filter1d(z, sigma)
    smoothed_streamline = np.asarray([x3, y3, z3], dtype=float).T

    # Ensure first and last point remain the same
    smoothed_streamline[0] = streamline[0]
    smoothed_streamline[-1] = streamline[-1]

    return smoothed_streamline


def smooth_line_spline(streamline, smoothing_parameter, nb_ctrl_points):
    """ Smooth a streamline using a spline. The number of control points
    can be specified, but must be at least 3.

    Parameters
    ----------
    streamline: np.ndarray
        The streamline to smooth.
    smoothing_parameter: float
        The sigma of the spline.
    nb_ctrl_points: int
        The number of control points.

    Returns
    -------
    smoothed_streamline: np.ndarray
        The smoothed streamline.
    """

    if smoothing_parameter < 0.00001:
        raise ValueError('Cant have a 0 sigma with spline.')

    if length(streamline) < 1:
        logging.debug('Streamline shorter than 1mm, corner cases possible.')

    if nb_ctrl_points < 3:
        nb_ctrl_points = 3

    initial_nb_of_points = len(streamline)

    # Resample the streamline to have the desired number of points
    # which will be used as control points for the spline
    sampled_streamline = set_number_of_points(streamline, nb_ctrl_points)

    # Fit the spline using the control points
    tck, u = splprep(sampled_streamline.T, s=smoothing_parameter)
    # Evaluate the spline
    smoothed_streamline = splev(np.linspace(0, 1, initial_nb_of_points), tck)
    smoothed_streamline = np.squeeze(np.asarray([smoothed_streamline]).T)

    # Ensure first and last point remain the same
    smoothed_streamline[0] = streamline[0]
    smoothed_streamline[-1] = streamline[-1]

    return smoothed_streamline
