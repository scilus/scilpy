# -*- coding: utf-8 -*-

import logging

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamlinespeed import (length, set_number_of_points)
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage.filters import gaussian_filter1d


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
    clusters = qbx_and_merge(orig_sft.streamlines, thresholds, nb_pts=20,
                             verbose=False)

    logging.debug("Done. Now getting list of indices in each of the {} "
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

    if sft.data_per_point is not None:
        logging.debug("Initial stateful tractogram contained data_per_point. "
                      "This information will not be carried in the final"
                      "tractogram.")
    new_sft = StatefulTractogram.from_sft(
        new_streamlines, sft, data_per_streamline=sft.data_per_streamline)

    return new_sft


def smooth_line_gaussian(streamline, sigma):
    if sigma < 0.00001:
        ValueError('Cant have a 0 sigma with gaussian.')

    nb_points = int(length(streamline))
    if nb_points < 2:
        logging.debug('Streamline shorter than 1mm, corner cases possible.')
        nb_points = 2
    sampled_streamline = set_number_of_points(streamline, nb_points)

    x, y, z = sampled_streamline.T
    x3 = gaussian_filter1d(x, sigma)
    y3 = gaussian_filter1d(y, sigma)
    z3 = gaussian_filter1d(z, sigma)
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


def get_theta(requested_theta, tracking_type):
    if requested_theta is not None:
        theta = requested_theta
    elif tracking_type == 'prob':
        theta = 20
    elif tracking_type == 'eudx':
        theta = 60
    else:
        theta = 45
    return theta


def sample_distribution(dist):
    """
    Parameters
    ----------
    dist: numpy.array
        The empirical distribution to sample from.

    Return
    ------
    ind: int
        The index of the sampled element.
    """
    cdf = dist.cumsum()
    if cdf[-1] == 0:
        return None
    return cdf.searchsorted(np.random.random() * cdf[-1])
