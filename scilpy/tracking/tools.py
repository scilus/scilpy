# -*- coding: utf-8 -*-

import logging

from dipy.tracking.metrics import downsample, length
from dipy.tracking.streamline import set_number_of_points
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage.filters import gaussian_filter1d


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
