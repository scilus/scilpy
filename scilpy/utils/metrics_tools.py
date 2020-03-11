#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
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


def get_bundle_metrics_meanstd(streamlines, metrics_files,
                               density_weighting=True):
    """
    Returns the mean value of each metric for the whole bundle, only
    considering voxels that are crossed by streamlines. The mean values are
    weighted by the number of streamlines crossing a voxel by default.
    If false, every voxel traversed by a streamline has the same weight.

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
        of a metric, and the second element is the standard deviation, for each
        metric.
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


def get_bundle_metrics_meanstdperpoint(streamlines, metrics_files,
                                       density_weighting=True):
    raise NotImplementedError


def plot_metrics_stats(mean, std, title=None, xlabel=None,
                       ylabel=None, figlabel=None, fill_color=None):
    """
    Plots the mean of a metric along n points with the standard deviation.

    Parameters
    ----------
    mean: Numpy 1D array of size n
        Mean of the metric along n points.
    std: Numpy 1D array of size n
        Standard deviation of the metric along n points.
    title: string
        Title of the figure.
    xlabel: string
        Label of the X axis.
    ylabel: string
        Label of the Y axis (suggestion: the metric name).
    figlabel: string
        Label of the figure (only metadata in the figure object returned).
    fill_color: string
        Hexadecimal RGB color filling the region between mean ± std. The
        hexadecimal RGB color should be formatted as #RRGGBB

    Return
    ------
    The figure object.
    """
    matplotlib.style.use('ggplot')

    fig, ax = plt.subplots()

    # Set optional information to the figure, if required.
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if figlabel is not None:
        fig.set_label(figlabel)

    dim = np.arange(1, len(mean)+1, 1)

    if len(mean) <= 20:
        ax.xaxis.set_ticks(dim)

    ax.set_xlim(0, len(mean)+1)

    # Plot the mean line.
    ax.plot(dim, mean, color="k", linewidth=5, solid_capstyle='round')

    # Plot the std
    plt.fill_between(dim, mean - std, mean + std, facecolor=fill_color)

    plt.close(fig)
    return fig
