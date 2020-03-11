#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def weighted_mean_stddev(weights, data):
    """
    Returns the weighted mean and standard deviation of the data.

    Parameters
    ------------
    weights : ndarray
        a ndarray containing the weighting factor

    data : ndarray
        the ndarray containing the data for which the stats are desired

    Returns
    ---------
    stats : tuple
        a tuple containing the mean and standard deviation of the data
    """

    mean = np.average(data, weights=weights)
    variance = np.average((data-mean)**2, weights=weights)

    return mean, np.sqrt(variance)


def get_metrics_stats_over_streamlines(streamlines, metrics_files,
                                       density_weighting=True):
    """
    Returns the mean value of each metric, only considering voxels that
    are crossed by streamlines. The mean values are weighted by the number of
    streamlines crossing a voxel by default. If false, every voxel traversed
    by a streamline has the same weight.

    Parameters
    ------------
    streamlines : sequence
        sequence of T streamlines. One streamline is an ndarray of shape
        (N, 3), where N is the number of points in that streamline, and
        ``streamlines[t][n]`` is the n-th point in the t-th streamline. Points
        are of form x, y, z in voxmm coordinates.

    metrics_files : sequence
        list of nibabel objects representing the metrics files

    density_weighting : bool
        weigh by the mean by the density of streamlines going through the voxel

    Returns
    ---------
    stats : list
        list of tuples where the first element of the tuple is the mean
        of a metric, and the second element is the standard deviation

    """

    # Compute weighting matrix taking the possible compression into account
    anat_dim = metrics_files[0].header.get_data_shape()
    weights = compute_tract_counts_map(streamlines, anat_dim)

    if not density_weighting:
        weights = weights > 0

    return map(lambda metric_file:
               weighted_mean_stddev(
                    weights,
                    metric_file.get_data().astype(np.float64)),
               metrics_files)
