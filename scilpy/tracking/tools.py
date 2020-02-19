#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

from dipy.tracking.metrics import downsample
from dipy.tracking.streamline import set_number_of_points
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.tracking.streamlinespeed import length


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
    filtered_sft = StatefulTractogram(filtered_streamlines, sft, Space.RASMM,
                                      data_per_point=filtered_data_per_point,
                                      data_per_streamline=filtered_data_per_streamline,
                                      origin=sft.origin)

    # Return to original space
    if orig_space == Space.VOX:
        filtered_sft.to_vox()
    elif orig_space == Space.VOXMM:
        filtered_sft.to_voxmm()

    return filtered_sft


def get_subset_streamlines(sft, max_streamlines, rng_seed=None):
    """
    Extract a specific number of streamlines

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
    ind = np.arange(sft._get_streamline_count)
    rng.shuffle(ind)

    subset_streamlines = list(np.asarray(sft.streamlines)[ind[:max_streamlines]])
                                                                                        # Calculated data_per_point and data_per_streamline
                                                                                        # Not needed to create a tractogram. But do we want to
                                                                                        # compute it manually?
    subset_sft = StatefulTractogram(subset_streamlines, sft, Space.RASMM,
                                    origin=sft.origin)

    return subset_sft


def resample_streamlines_num_points(streamlines, num_points=0, arc_length=False):
    """
    Resample streamlines using number of points per streamline

    Parameters
    ----------
    streamlines: list
        List of list of 3D points. Ex, if working with StatefulTractograms:
        sft.streamlines.
    num_points: int
        Number of points per streamline in the output.
    arc_length: bool
        Whether to downsample using arc length parametrization.

    Return
    ------
    resampled_streamlines: list
        List of resampled streamlines.
    """
    resampled_streamlines = []
    for streamline in streamlines:
        if arc_length:
            line = set_number_of_points(streamline, num_points)
        else:
            line = downsample(streamline, num_points)
        resampled_streamlines.append(line)

    return resampled_streamlines


def resample_streamlines_step_size(streamlines, step_size=0, arc_length=False):
    """
    Resample streamlines using a fixed step size.

    Parameters
    ----------
    streamlines: list
        List of list of 3D points. Ex, if working with StatefulTractograms:
        sft.streamlines.
    num_points: int
        Number of points per streamline in the output.
    arc_length: bool
        Whether to downsample using arc length parametrization.

    Return
    ------
    resampled_streamlines: list
        List of resampled streamlines.
    """
    resampled_streamlines = []
    for streamline in streamlines:
        if arc_length:
            line = set_number_of_points(streamline, num_points)
        else:
            line = downsample(streamline, num_points)
        resampled_streamlines.append(line)

    return resampled_streamlines


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
