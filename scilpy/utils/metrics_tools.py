# -*- coding: utf-8 -*-

import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii


def get_bundle_metrics_profiles(sft, metrics_files):
    """
    Returns the profile of each metric along each streamline from a sft.
    This is used to create tract profiles.

    Parameters
    ------------
    sft : StatefulTractogram
        Input bundle under which to compute profile.
    metrics_files : sequence
        list of nibabel objects representing the metrics files

    Returns
    ---------
    profiles_values : list
        list of profiles for each streamline, per metric given

    """

    sft.to_vox()
    sft.to_corner()
    streamlines = sft.streamlines

    def _get_profile_one_streamline(streamline, metrics_files):
        x_ind = np.floor(streamline[:, 0]).astype(np.int)
        y_ind = np.floor(streamline[:, 1]).astype(np.int)
        z_ind = np.floor(streamline[:, 2]).astype(np.int)

        return list(map(lambda metric_file: metric_file[x_ind, y_ind, z_ind],
                    metrics_files))

    # We preload the data to avoid loading it for each streamline
    metrics_data = list(map(lambda metric_file: metric_file.get_fdata(),
                        metrics_files))

    # The root list has S elements, where S == the number of streamlines.
    # Each element from S is a sublist with N elements, where N is the number
    # of metrics. Each element from N is a list of the metric values
    # encountered along the current streamline.
    metrics_per_strl =\
        list(map(lambda strl: _get_profile_one_streamline(strl, metrics_data),
             streamlines))

    converted = []
    # Here, the zip gives us a list of N tuples, so one tuple for each metric.
    # Each tuple has S elements, where S is the number of streamlines.
    # We then convert each tuple to a numpy array
    for metric_values in zip(*metrics_per_strl):
        converted.append(np.asarray(metric_values))

    return converted


def weighted_mean_std(weights, data):
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


def get_bundle_metrics_mean_std(streamlines, metrics_files,
                                density_weighting=True):
    """
    Returns the mean value of each metric for the whole bundle, only
    considering voxels that are crossed by streamlines. The mean values are
    weighted by the number of streamlines crossing a voxel by default.
    If false, every voxel traversed by a streamline has the same weight.

    Parameters
    ------------
    streamlines : list of numpy.ndarray
        Input streamlines under which to compute stats.
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
               weighted_mean_std(weights,
                                 metric_file.get_fdata(dtype=np.float64)),
               metrics_files)


def get_bundle_metrics_mean_std_per_point(streamlines, bundle_name,
                                          distances_to_centroid_streamline,
                                          metrics, labels, density_weighting=False,
                                          distance_weighting=False):
    """
    Compute the mean and std PER POiNT of the bundle for every given metric.

    Parameters
    ----------
    streamlines: list of numpy.ndarray
        Input streamlines under which to compute stats.
    bundle_name: str
        Name of the bundle. Will be used as a key in the dictionary.
    distances_to_centroid_streamline: np.ndarray
        List of distances obtained with scil_label_and_distance_maps.py
    metrics: sequence
        list of nibabel objects representing the metrics files
    labels: np.ndarray
        List of labels obtained with scil_label_and_distance_maps.py
    density_weighting: bool
        If true, weight statistics by the number of streamlines passing through
        each voxel. [False]
    distance_weighting: bool
        If true, weight statistics by the inverse of the distance between a
        streamline and the centroid.

    Returns
    -------
    stats
    """
    # Computing infos on bundle
    unique_labels = np.unique(labels)
    num_digits_labels = len(str(np.max(unique_labels)))
    if density_weighting:
        track_count = compute_tract_counts_map(streamlines,
                                               metrics[0].shape).astype(np.float64)
    else:
        track_count = np.ones(metrics[0].shape)

    # Bigger weight near the centroid streamline
    distances_to_centroid_streamline = 1.0 / distances_to_centroid_streamline

    # Keep data as int to get the underlying voxel
    bundle_data_int = streamlines.data.astype(np.int)

    # Get stats
    stats = {bundle_name: {}}
    for metric in metrics:
        metric_data = metric.get_fdata(dtype=np.float64)
        current_metric_fname, _ = split_name_with_nii(
            os.path.basename(metric.get_filename()))
        stats[bundle_name][current_metric_fname] = {}

        for i in unique_labels:
            number_key = '{}'.format(i).zfill(num_digits_labels)
            label_stats = {}
            stats[bundle_name][current_metric_fname][number_key] = label_stats

            label_indices = bundle_data_int[labels == i]
            label_metric = metric_data[label_indices[:, 0],
                                       label_indices[:, 1],
                                       label_indices[:, 2]]
            track_weight = track_count[label_indices[:, 0],
                                       label_indices[:, 1],
                                       label_indices[:, 2]]
            label_weight = track_weight
            if distance_weighting:
                label_weight *= distances_to_centroid_streamline[labels == i]
            if np.sum(label_weight) == 0:
                logging.warning('Weights sum to zero, can\'t be normalized. '
                                'Disabling weighting')
                label_weight = None

            label_mean = np.average(label_metric,
                                    weights=label_weight)
            label_std = np.sqrt(np.average(
                (label_metric - label_mean) ** 2,
                weights=label_weight))
            label_stats['mean'] = float(label_mean)
            label_stats['std'] = float(label_std)
    return stats


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


def get_roi_metrics_mean_std(density_map, metrics_files):
    """
    Returns the mean and standard deviation of each metric, using the
    provided density map. This can be a binary mask,
    or contain weighted values between 0 and 1.

    Parameters
    ------------
    density_map : ndarray
        3D numpy array containing a density map.
    metrics_files : sequence
        list of nibabel objects representing the metrics files.

    Returns
    ---------
    stats : list
        list of tuples where the first element of the tuple is the mean
        of a metric, and the second element is the standard deviation.

    """

    return map(lambda metric_file:
               weighted_mean_std(density_map,
                                 metric_file.get_fdata()),
               metrics_files)
