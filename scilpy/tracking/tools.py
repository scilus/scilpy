# -*- coding: utf-8 -*-

import logging

from dipy.io.stateful_tractogram import StatefulTractogram
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

    # Compute streamlines lengths
    lengths = length(sft.streamlines)

    # Filter lengths
    filter_stream = np.logical_and(lengths >= min_length,
                                   lengths <= max_length)
    filtered_streamlines = list(np.asarray(sft.streamlines)[filter_stream])
    filtered_data_per_point = sft.data_per_point[filter_stream]
    filtered_data_per_streamline = sft.data_per_streamline[filter_stream]

    # Create final sft
    filtered_sft = StatefulTractogram.from_sft(
        filtered_streamlines, sft,
        data_per_point=filtered_data_per_point,
        data_per_streamline=filtered_data_per_streamline)

    # Return to original space
    filtered_sft.to_space(orig_space)

    return filtered_sft


def get_subset_streamlines(sft, max_streamlines, rng_seed=None):
    """
    Extract a specific number of streamlines.

    Parameters
    ----------
    sft: StatefulTractogram
        SFT containing the streamlines to subsample.
    max_streamlines: int
        Maximum number of streamlines to output.
    rng_seed: int
        Random number to use for shuffling the data.

    Return
    ------
    subset_sft: StatefulTractogram
        The filtered streamlines as a sft.
    """

    rng = np.random.RandomState(rng_seed)
    ind = np.arange(len(sft.streamlines))
    rng.shuffle(ind)

    subset_streamlines = list(np.asarray(sft.streamlines)[
                              ind[:max_streamlines]])
    subset_data_per_point = sft.data_per_point[ind[:max_streamlines]]
    subset_data_per_streamline = sft.data_per_streamline[ind[:max_streamlines]]

    subset_sft = StatefulTractogram.from_sft(
        subset_streamlines, sft,
        data_per_point=subset_data_per_point,
        data_per_streamline=subset_data_per_streamline)

    return subset_sft


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
    smoothed_streamline = np.asarray([x3, y3, z3]).T

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
