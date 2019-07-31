from __future__ import division

import dipy.tracking.metrics
import dipy.tracking.utils
import logging
import numpy as np

from dipy.tracking.metrics import downsample
from dipy.tracking.streamline import set_number_of_points


def filter_streamlines_by_length(streamlines,
                                 data_per_point,
                                 data_per_streamline,
                                 min_length=0., max_length=np.inf):
    """
    Parameters
    ----------
    streamlines: list
        List of list of 3D points.

    data_per_point: dict

    data_per_streamline: dict

    min_length: float
        Minimum length of streamlines.
    max_length: float
        Maximum length of streamlines.

    Return
    ------
    new_streamlines: list
        List of filtered streamlines by length.

    new_data: dict
        data_per_point
        data_per_streamline
    """

    num_streamlines = len(streamlines)

    lengths = np.zeros(num_streamlines)

    for i in np.arange(num_streamlines):
        lengths[i] = dipy.tracking.metrics.length(streamlines[i])

    filterStream = np.logical_and(lengths>=min_length, lengths<=max_length)
    per_point = data_per_point[filterStream]
    per_streamline = data_per_streamline[filterStream]

    data = {per_point: per_point,
            per_streamline: per_streamline}

    return list(np.asarray(streamlines)[filterStream]), data

def get_subset_streamlines(streamlines,
                           data_per_point,
                           data_per_streamline,
                           max_streamlines, rng=None):
    """
    Parameters
    ----------
    streamlines: list
        List of list of 3D points.
    max_streamlines: int
        Maximum number of streamlines to output.
    rng: RandomState object
        Random number generator to use for shuffling the data.
        By default, a constant seed is used.

    Return
    ------
    average: list
        List of subsampled streamlines.
    """

    if rng is None:
        rng = np.random.RandomState(1234)

    ind = np.arange(len(streamlines))
    rng.shuffle(ind)

    per_point = data_per_point[ind[:max_streamlines]]
    per_streamline = data_per_streamline[ind[:max_streamlines]]

    data = {'per_point': per_point,
            'per_streamline': per_streamline}

    return list(np.asarray(streamlines)[ind[:max_streamlines]]), data

def resample_streamlines(streamlines, num_points=0, arc_length=False):
    """
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
    average: list
        List of subsampled streamlines.
    """

    for i in range(len(streamlines)):
        if arc_length:
            line = set_number_of_points(streamlines[i], num_points)
        else:
            line = downsample(streamlines[i], num_points)
        results.append(line)

    return results
