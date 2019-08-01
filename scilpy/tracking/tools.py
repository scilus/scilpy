from __future__ import division

from dipy.tracking.metrics import length, downsample
from dipy.tracking.streamline import set_number_of_points
import dipy.tracking.utils
import numpy as np


def filter_streamlines_by_length(streamlines,
                                 data_per_point,
                                 data_per_streamline,
                                 min_length=0., max_length=np.inf):
    """
    Filter streamlines using minimum and max length

    Parameters
    ----------
    streamlines: list
        List of list of 3D points.

    data_per_point: dict
        dict of metrics with one value per point per streamline
    data_per_streamline: dict
        dict of metrics with one value per streamline

    min_length: float
        Minimum length of streamlines.
    max_length: float
        Maximum length of streamlines.

    Return
    ------
    filtered_streamlines: list
        List of filtered streamlines by length.

    filtered_per_point: dict
        dict of metrics per point for filtered streamlines
    filtered_per_streamline: dict
            dict of metrics per streamline for filtered streamlines
    """

    lengths = []
    for streamline in streamlines:
        lengths.append(lengths(streamline))

    lengths = np.asarray(lengths)

    filter_stream = np.logical_and(lengths >= min_length, lengths <= max_length)

    filtered_streamlines = list(np.asarray(streamlines)[filter_stream])
    filtered_data_per_point = data_per_point[filter_stream]
    filtered_data_per_streamline = data_per_streamline[filter_stream]

    return filtered_streamlines, filtered_per_point, filtered_per_streamline


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
        dict of metrics with one value per point per streamline
    data_per_streamline: dict
        dict of metrics with one value per streamline

    max_streamlines: int
        Maximum number of streamlines to output.
    rng: RandomState object
        Random number generator to use for shuffling the data.
        By default, a constant seed is used.

    Return
    ------
    subset_streamlines: list
        List of a subset streamline.

    subset_per_point: dict
        dict of metrics per point for filtered streamlines
    subset_per_streamline: dict
            dict of metrics per streamline for filtered streamlines
    """

    if rng_seed is None:
        rng_seed = np.random.RandomState(1234)
    else:
        rng_seed = np.random.RandomState(rng_seed)

    ind = np.arange(len(streamlines))
    rng_seed.shuffle(ind)

    subset_streamlines = list(np.asarray(streamlines)[ind[:max_streamlines]])
    subset_per_point = data_per_point[ind[:max_streamlines]]
    subset_per_streamline = data_per_streamline[ind[:max_streamlines]]

    return subset_streamlines, subset_per_point, subset_streamlines


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
    for streamline in in streamlines:
        if arc_length:
            line = set_number_of_points(streamline, num_points)
        else:
            line = downsample(streamline, num_points)
        resampled_streamlines.append(line)

    return resampled_streamlines
