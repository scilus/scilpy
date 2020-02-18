#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

from dipy.tracking.metrics import length, downsample
from dipy.tracking.streamline import set_number_of_points
from dipy.io.stateful_tractogram import Space, StatefulTractogram

                                                                #from dipy.tracking.streamlinespeed import length # Philippe utilisait ce length. Différent?


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
    filtered_streamlines: list
        List of filtered streamlines by length.
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

    # Create final sft
    filtered_sft = StatefulTractogram(filtered_streamlines, sft, Space.RASMM,
                                  origin=sft.origin)                             # ??? Philippe utilisait shifterd_origin. A pas l'air d'exister...

    # Return to original space
    if orig_space == Space.VOX:
        filtered_sft.to_vox()
    elif orig_space == Space.VOXMM:
        filtered_sft.to_voxmm()

    return filtered_sft, filtered_streamlines


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
