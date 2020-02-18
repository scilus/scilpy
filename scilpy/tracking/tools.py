#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

from dipy.tracking.metrics import length, downsample
from dipy.tracking.streamline import set_number_of_points
from dipy.io.stateful_tractogram import Space, StatefulTractogram


def filter_streamlines_by_length(sft, min_length=0., max_length=np.inf,
                                 return_details=True):
    """
    Filter streamlines using minimum and max length

    Parameters
    ----------
    sft: StatefulTractogram
        SFT containing the streamlines to filter.
    min_length: float
        Minimum length of streamlines, in the same space as the sft.
    max_length: float
        Maximum length of streamlines, in the same space as the sft.
    return_details: bool
        If true, returns the filtered streamline, data_per_point and
        data_per_streamlines together with the sft.

    Return
    ------
    good_sft : StatefulTractogram
        A tractogram without short streamlines.
    (optionally:)
    good_streamlines: list
        List of filtered streamlines by length.
    good_data_per_point: dict
        dict of data per point for filtered streamlines.
    good_data_per_streamline: dict
        dict of data per streamline for filtered streamlines.
    """

    lengths = []
    for streamline in sft.streamlines:
        lengths.append(length(streamline))

    lengths = np.asarray(lengths)

    filter_stream = np.logical_and(lengths >= min_length,
                                   lengths <= max_length)

    good_streamlines = list(np.asarray(sft.streamlines)[filter_stream])
    good_data_per_point = sft.data_per_point[filter_stream]
    good_data_per_streamline = sft.data_per_streamline[filter_stream]

    good_sft = StatefulTractogram(good_streamlines, sft, Space.RASMM,
                                  data_per_streamline=good_data_per_streamline,
                                  data_per_point=good_data_per_point)

    if return_details:
        return good_sft, good_streamlines, good_data_per_point, \
               good_data_per_streamline
    else:
        return good_sft


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


def resample_streamlines(streamlines, num_points=0, arc_length=False):
    """
    Resample streamlines using number of points per streamline

    Parameters
    ----------
    streamlines: list
        List of list of 3D points.
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
