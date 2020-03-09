#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import logging

import numpy as np

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import (length, set_number_of_points)


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
        List of a subset streamline.
    """

    rng = np.random.RandomState(rng_seed)
    ind = np.arange(len(sft.streamlines))
    rng.shuffle(ind)

    subset_streamlines = list(np.asarray(sft.streamlines)[ind[:max_streamlines]])
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
