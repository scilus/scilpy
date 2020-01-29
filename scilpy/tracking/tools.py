#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

from dipy.tracking.metrics import length, downsample
from dipy.tracking.streamline import set_number_of_points


def filter_streamlines_by_length(sft, min_length=0., max_length=np.inf):
    """
    Filter streamlines using minimum and max length

    Parameters
    ----------
    streamlines: list
        List of list of 3D points.

    data_per_point: dict
        dict of data with one value per point per streamline
    data_per_streamline: dict
        dict of data with one value per streamline

    min_length: float
        Minimum length of streamlines.
    max_length: float
        Maximum length of streamlines.

    Return
    ------
    filtered_streamlines: list
        List of filtered streamlines by length.

    filtered_data_per_point: dict
        dict of data per point for filtered streamlines.
    filtered_data_per_streamline: dict
        dict of data per streamline for filtered streamlines.
    """

    lengths = []
    for streamline in sft.streamlines:
        lengths.append(length(streamline))

    lengths = np.asarray(lengths)

    filter_stream = np.logical_and(lengths >= min_length,
                                   lengths <= max_length)

    filtered_streamlines = list(np.asarray(sft.streamlines)[filter_stream])
    filtered_data_per_point = sft.data_per_point[filter_stream]
    filtered_data_per_streamline = sft.data_per_streamline[filter_stream]

    return filtered_streamlines, filtered_data_per_point, filtered_data_per_streamline


def get_subset_streamlines(streamlines,
                           data_per_point,
                           data_per_streamline,
                           max_streamlines, rng_seed=None):
    """
    Extract a specific number of streamlines

    Parameters
    ----------
    streamlines: list
        List of list of 3D points.

    data_per_point: dict
        dict of data with one value per point per streamline
    data_per_streamline: dict
        dict of data with one value per streamline

    max_streamlines: int
        Maximum number of streamlines to output.
    rng_seed: int
        Random number to use for shuffling the data.

    Return
    ------
    subset_streamlines: list
        List of a subset streamline.

    subset_data_per_point: dict
        dict of data per point for subset of streamlines
    subset_data_per_streamline: dict
        dict of data per streamline for subset of streamlines
    """

    rng = np.random.RandomState(rng_seed)
    ind = np.arange(len(streamlines))
    rng.shuffle(ind)

    subset_streamlines = list(np.asarray(streamlines)[ind[:max_streamlines]])
    subset_data_per_point = data_per_point[ind[:max_streamlines]]
    subset_data_per_streamline = data_per_streamline[ind[:max_streamlines]]

    return subset_streamlines, subset_data_per_point, subset_data_per_streamline


def resample_streamlines(streamlines, num_points=0, step_size=0,
                         arc_length=False):
    """
    Resample streamlines using number of points per streamline

    Parameters
    ----------
    streamlines: list
        List of list of 3D points.
    num_points: int
        Number of points per streamline in the output.
    OR
    step_size: float
        Step size that all streamlines should have.
    arc_length: bool
        Whether to downsample using arc length parametrization.

    Return
    ------
    resampled_streamlines: list
        List of resampled streamlines.
    """
    resampled_streamlines = []
    if step_size>0 and num_points>0:
        raise ValueError("Please choose only one method. num_points OR "
                         "step_size")
    elif num_points > 0:
        for streamline in streamlines:
            if arc_length:
                line = set_number_of_points(streamline, num_points)
            else:
                line = downsample(streamline, num_points)
            resampled_streamlines.append(line)
    else:  # step_size>0
        lengths = length(streamlines)
        nb_points = np.ceil(lengths / step_size).astype(int)
        resampled_streamlines = [set_number_of_points(s, n) for s, n in
                                 zip(streamlines, nb_points)]
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
