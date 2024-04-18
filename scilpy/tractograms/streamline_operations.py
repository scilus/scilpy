# -*- coding: utf-8 -*-
import copy
import logging
from multiprocessing import Pool

import numpy as np
import scipy.ndimage as ndi
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking import metrics as tm
from dipy.tracking.streamlinespeed import (compress_streamlines,
                                           length,
                                           set_number_of_points)
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation


def _get_streamline_pt_index(points_to_index, vox_index, from_start=True):
    """Get the index of the streamline point in the voxel.

    Parameters
    ----------
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


def get_angles(sft):
    """Color streamlines according to their length.

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram.

    Returns
    -------
    angles: list[np.ndarray]
        The angles per streamline, in degree.
    """
    angles = []
    for i in range(len(sft.streamlines)):
        dirs = np.diff(sft.streamlines[i], axis=0)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        cos_angles = np.sum(dirs[:-1, :] * dirs[1:, :], axis=1)
        # Resolve numerical instability
        cos_angles = np.minimum(np.maximum(-1.0, cos_angles), 1.0)
        line_angles = [0.0] + list(np.arccos(cos_angles)) + [0.0]
        angles.extend(line_angles)

    angles = np.rad2deg(angles)

    return angles


def get_values_along_length(sft):
    """Get the streamlines' coordinate positions according to their length.

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram that contains the list of streamlines to be colored

    Returns
    -------
    positions: list[list]
        For each streamline, the linear distribution of its length.
    """
    positions = []
    for i in range(len(sft.streamlines)):
        positions.extend(list(np.linspace(0, 1, len(sft.streamlines[i]))))

    return positions


def compress_sft(sft, tol_error=0.01):
    """
    Compress a stateful tractogram. Uses Dipy's compress_streamlines, but
    deals with space better.

    Dipy's description:
    The compression consists in merging consecutive segments that are
    nearly collinear. The merging is achieved by removing the point the two
    segments have in common.

    The linearization process [Presseau15] ensures that every point being
    removed are within a certain margin (in mm) of the resulting streamline.
    Recommendations for setting this margin can be found in [Presseau15]
    (in which they called it tolerance error).

    The compression also ensures that two consecutive points won't be too far
    from each other (precisely less or equal than *max_segment_length* mm).
    This is a tradeoff to speed up the linearization process [Rheault15]. A
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
    compressed_sft: StatefulTractogram
    """
    # Go to world space
    orig_space = sft.space
    sft.to_rasmm()

    # Compress streamlines
    compressed_streamlines = compress_streamlines(sft.streamlines,
                                                  tol_error=tol_error)
    if sft.data_per_point is not None and sft.data_per_point.keys():
        logging.warning("Initial StatefulTractogram contained data_per_point. "
                        "This information will not be carried in the final "
                        "tractogram.")

    compressed_sft = StatefulTractogram.from_sft(
        compressed_streamlines, sft,
        data_per_streamline=sft.data_per_streamline)

    # Return to original space
    compressed_sft.to_space(orig_space)

    return compressed_sft


def cut_invalid_streamlines(sft, epsilon=0.001):
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
                        sft.data_per_point[key][ind][
                            best_pos[0]:best_pos[1]-1])
            else:
                logging.warning('Streamlines entirely out of the volume.')
        else:
            new_streamlines.append(sft.streamlines[ind])
            for key in sft.data_per_streamline.keys():
                new_data_per_streamline[key].append(
                    sft.data_per_streamline[key][ind])
            for key in sft.data_per_point.keys():
                new_data_per_point[key].append(sft.data_per_point[key][ind])
    new_sft = StatefulTractogram.from_sft(
        new_streamlines, sft, data_per_streamline=new_data_per_streamline,
        data_per_point=new_data_per_point)

    # Move the streamlines back to the original space/origin
    sft.to_space(space)
    sft.to_origin(origin)

    new_sft.to_space(space)
    new_sft.to_origin(origin)

    return new_sft, cutting_counter


def remove_single_point_streamlines(sft):
    """
    Remove single point streamlines from a StatefulTractogram.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to remove single point streamlines from.

    Returns
    -------
    new_sft : StatefulTractogram
        New object with the single point streamlines removed.
    """
    indices = [i for i in range(len(sft)) if len(sft.streamlines[i]) > 1]
    if len(indices):
        new_sft = sft[indices]
    else:
        new_sft = StatefulTractogram.from_sft([], sft)

    return new_sft


def remove_overlapping_points_streamlines(sft, threshold=0.001):
    """
    Remove overlapping points from streamlines in a StatefulTractogram.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to remove overlapping points from.
    threshold: float (optional)
        Maximum distance between two points to be considered overlapping.
        Default: 0.001 mm.

    Returns
    -------
    new_sft : StatefulTractogram
        New object with the overlapping points removed from each streamline.
    """
    new_streamlines = []
    for streamline in sft.streamlines:
        norm = np.linalg.norm(np.diff(streamline, axis=0),
                              axis=1)

        indices = np.where(norm < threshold)[0]
        if len(indices) == 0:
            new_streamlines.append(streamline.tolist())
        else:
            new_streamline = np.delete(streamline.tolist(),
                                       indices, axis=0)
            new_streamlines.append(new_streamline)
    new_sft = StatefulTractogram.from_sft(new_streamlines, sft)

    return new_sft


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

    logging.info("Total length per orientation is:\n"
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
        logging.info("The value of your step size seems suspiciously low. "
                     "Please check.")
    elif step_size > np.max(sft.voxel_sizes):
        logging.info("The value of your step size seems suspiciously high. "
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
        logging.info("Initial StatefulTractogram contained data_per_point. "
                     "This information will not be carried in the final "
                     "tractogram.")
    new_sft = StatefulTractogram.from_sft(
        new_streamlines, sft, data_per_streamline=sft.data_per_streamline)

    return new_sft


def smooth_line_gaussian(streamline, sigma):
    """
    Smooths a streamline using a gaussian filter. Enforces the endpoints to
    remain the same.

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
        logging.info('Streamline shorter than 1mm, corner cases possible.')

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
    """
    Smooths a streamline using a spline. The number of control points can be
    specified, but must be at least 3. The final streamline will have the same
    number of points as the input streamline. Enforces the endpoints to remain
    the same.

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
        logging.info('Streamline shorter than 1mm, corner cases possible.')

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


def generate_matched_points(sft):
    """
    Generates an array where each element i is set to the index of the
    streamline to which it belongs

    Parameters:
    -----------
    sft : StatefulTractogram
        The stateful tractogram containing the streamlines.

    Returns:
    --------
    matched_points : ndarray
        An array where each element is set to the index of the streamline
        to which it belongs
    """
    tmp_len = [len(s) for s in sft.streamlines]
    total_points = np.sum(tmp_len)
    offsets = np.insert(np.cumsum(tmp_len), 0, 0)

    matched_points = np.zeros(total_points, dtype=np.uint64)

    for i in range(len(offsets) - 1):
        matched_points[offsets[i]:offsets[i+1]] = i

    matched_points[offsets[-1]:] = len(offsets) - 1

    return matched_points


def parallel_transport_streamline(streamline, nb_streamlines, radius,
                                  rng=None):
    """ Generate new streamlines by parallel transport of the input
    streamline. See [0] and [1] for more details.

    [0]: Hanson, A.J., & Ma, H. (1995). Parallel Transport Approach to 
        Curve Framing. # noqa E501
    [1]: TD Essentials: Parallel Transport.
        https://www.youtube.com/watch?v=5LedteSEgOE

    Parameters
    ----------
    streamline: ndarray (N, 3)
        The streamline to transport.
    nb_streamlines: int
        The number of streamlines to generate.
    radius: float
        The radius of the circle around the original streamline in which the
        new streamlines will be generated.
    rng: numpy.random.Generator, optional
        The random number generator to use. If None, the default numpy
        random number generator will be used.

    Returns
    -------
    new_streamlines: list of ndarray (N, 3)
        The generated streamlines.
    """

    if rng is None:
        rng = np.random.default_rng(0)

    # Compute the tangent at each point of the streamline
    T = np.gradient(streamline, axis=0)
    # Normalize the tangents
    T = T / np.linalg.norm(T, axis=1)[:, None]

    # Placeholder for the normal vector at each point
    V = np.zeros_like(T)
    # Set the normal vector at the first point to kind of perpendicular to
    # the first direction vector
    V[0] = np.roll(streamline[0] - streamline[1], 1)
    V[0] = V[0] / np.linalg.norm(V[0])
    # For each point
    for i in range(0, T.shape[0] - 1):
        # Compute the torsion vector
        B = np.cross(T[i], T[i+1])
        # If the torsion vector is 0, the normal vector does not change
        if np.linalg.norm(B) < 1e-3:
            V[i+1] = V[i]
        # Else, the normal vector is rotated around the torsion vector by
        # the torsion.
        else:
            B = B / np.linalg.norm(B)
            theta = np.arccos(np.dot(T[i], T[i+1]))
            # Rotate the vector V[i] around the vector B by theta
            # radians.
            rotation = Rotation.from_rotvec(B * theta).as_matrix()
            V[i+1] = np.dot(rotation, V[i])

    # Compute the binormal vector at each point
    W = np.cross(T, V, axis=1)

    # Generate the new streamlines
    # TODO?: This could easily be optimized to avoid the for loop, we have to
    # see if this becomes a bottleneck.
    new_streamlines = []
    for i in range(nb_streamlines):
        # Get a random number between -1 and 1
        rand_v = rng.uniform(-1, 1)
        rand_w = rng.uniform(-1, 1)

        # Compute the norm of the "displacement"
        norm = np.sqrt(rand_v**2 + rand_w**2)
        # Displace the normal and binormal vectors by a random amount
        V_mod = V * rand_v
        W_mod = W * rand_w
        # Compute the displacement vector
        VW = (V_mod + W_mod)
        # Displace the streamline around the original one following the
        # parallel frame. Make sure to normalize the displacement vector
        # so that the new streamline is in a circle around the original one.

        new_s = streamline + (rng.uniform(0, 1) * VW / norm) * radius
        new_streamlines.append(new_s)

    return new_streamlines


def remove_loops_and_sharp_turns(streamlines, max_angle, qb_threshold=None,
                                 qb_seed=0, num_processes=1):
    """
    Remove loops and sharp turns from a list of streamlines.

    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which to remove loops and sharp turns.
    max_angle: float
        Maximal winding angle a streamline can have before
        being classified as a loop.
    qb_threshold: float, optional
        If not None, do the additional QuickBundles pass. This will help remove
        sharp turns. Should only be used on bundled streamlines, not on
        whole-brain tractograms. If set, value is Quickbundles distance
        threshold. Suggested default: 15.
    qb_seed: int
        Seed to initialize randomness in QuickBundles
    num_processes : int
        Split the calculation to a pool of children processes.

    Returns
    -------
    list: the ids of clean streamlines
        Only the ids are returned so proper filtering can be done afterwards
    """
    pool = Pool(num_processes)
    windings = pool.map(tm.winding, streamlines)
    pool.close()

    streamlines_clean = streamlines[np.array(windings) < max_angle]
    ids = list(np.where(np.array(windings) < max_angle)[0])

    if qb_threshold is not None:
        ids = []
        if len(streamlines_clean) > 1:
            curvature = []

            rng = np.random.RandomState(qb_seed)
            clusters = qbx_and_merge(streamlines_clean,
                                     [40, 30, 20, qb_threshold],
                                     rng=rng, verbose=False)

            for cc in clusters.centroids:
                curvature.append(tm.mean_curvature(cc))
            mean_curvature = sum(curvature)/len(curvature)

            for i in range(len(clusters.centroids)):
                if tm.mean_curvature(clusters.centroids[i]) <= mean_curvature:
                    ids.extend(clusters[i].indices)
        else:
            logging.info("Impossible to use the use_qb option because " +
                         "not more than one streamline left from the\n" +
                         "input file.")
    return ids


def get_streamlines_bounding_box(streamlines):
    """
    Classify inliers and outliers from a list of streamlines.

    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which inliers and outliers are separated.

    Returns
    -------
    tuple: Minimum and maximum corner coordinate of the streamlines
        bounding box
    """
    box_min = np.array([np.inf, np.inf, np.inf])
    box_max = -np.array([np.inf, np.inf, np.inf])

    for s in streamlines:
        box_min = np.minimum(box_min, np.min(s, axis=0))
        box_max = np.maximum(box_max, np.max(s, axis=0))

    return box_min, box_max
