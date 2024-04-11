# -*- coding: utf-8 -*-

import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii


def compute_lesion_stats(map_data, lesion_atlas, single_label=True,
                         voxel_sizes=[1.0, 1.0, 1.0], min_lesion_vol=7,
                         precomputed_lesion_labels=None):
    """
    Returns information related to lesion inside of a binary mask or voxel
    labels map (bundle, for tractometry).

    Parameters
    ------------
    map_data : np.ndarray
        Either a binary mask (uint8) or a voxel labels map (int16).
    lesion_atlas : np.ndarray (3)
        Labelled atlas of lesion. Should be int16.
    single_label : boolean
        If true, does not add an extra layer for number of labels.
    voxel_sizes : np.ndarray (3)
        If not specified, returns voxel count (instead of  volume)
    min_lesion_vol : float
        Minimum lesion volume in mm3 (default: 7, cross-shape).
    precomputed_lesion_labels : np.ndarray (N)
        For connectivity analysis, when the unique lesion labels are known,
        provided a pre-computed list of labels save computation.
    Returns
    ---------
    lesion_load_dict : dict
        For each label, volume and lesion count
    """
    voxel_vol = np.prod(voxel_sizes)

    if single_label:
        labels_list = [1]
    else:
        labels_list = np.unique(map_data)[1:].astype(np.int32)

    section_dict = {'lesion_total_volume': {}, 'lesion_volume': {},
                    'lesion_count': {}}
    for label in labels_list:
        zlabel = str(label).zfill(3)
        if not single_label:
            tmp_mask = np.zeros(map_data.shape, dtype=np.int16)
            tmp_mask[map_data == label] = 1
            tmp_mask *= lesion_atlas
        else:
            tmp_mask = lesion_atlas * map_data

        lesion_vols = []
        if precomputed_lesion_labels is None:
            computed_lesion_labels = np.unique(tmp_mask)[1:]
        else:
            computed_lesion_labels = precomputed_lesion_labels

        for lesion in computed_lesion_labels:
            curr_vol = np.count_nonzero(tmp_mask[tmp_mask == lesion]) \
                * voxel_vol
            if curr_vol >= min_lesion_vol:
                lesion_vols.append(curr_vol)
        if lesion_vols:
            section_dict['lesion_total_volume'][zlabel] = round(
                np.sum(lesion_vols), 3)
            section_dict['lesion_volume'][zlabel] = np.round(
                lesion_vols, 3).tolist()
            section_dict['lesion_count'][zlabel] = float(len(lesion_vols))
        else:
            section_dict['lesion_total_volume'][zlabel] = 0.0
            section_dict['lesion_volume'][zlabel] = [0.0]
            section_dict['lesion_count'][zlabel] = 0.0

    if single_label:
        section_dict = {'lesion_total_volume': section_dict['lesion_total_volume']['001'],
                        'lesion_volume': section_dict['lesion_volume']['001'],
                        'lesion_count': section_dict['lesion_count']['001']}

    return section_dict


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
        x_ind = np.floor(streamline[:, 0]).astype(int)
        y_ind = np.floor(streamline[:, 1]).astype(int)
        z_ind = np.floor(streamline[:, 2]).astype(int)

        return list(map(lambda metric_file: metric_file[x_ind, y_ind, z_ind],
                        metrics_files))

    # We preload the data to avoid loading it for each streamline
    metrics_data = list(map(lambda metric_file: metric_file.get_fdata(
        dtype=np.float64),
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
        converted.append(np.asarray(metric_values, dtype=float))

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

    masked_data = np.ma.masked_array(data, np.logical_or(np.isnan(data),
                                                         np.isinf(data)))
    mean = np.ma.average(masked_data, weights=weights)
    variance = np.ma.average((masked_data-mean)**2, weights=weights)

    return mean, np.sqrt(variance)


def get_bundle_metrics_mean_std(streamlines, metrics_files,
                                distance_values, correlation_values,
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
    weights = compute_tract_counts_map(streamlines, anat_dim).astype(float)

    if not density_weighting:
        weights[weights > 0] = 1

    if distance_values is not None:
        weights *= distance_values

    if correlation_values is not None:
        weights *= correlation_values

    return map(lambda metric_file:
               weighted_mean_std(weights,
                                 metric_file.get_fdata(dtype=np.float64)),
               metrics_files)


def get_bundle_metrics_mean_std_per_point(streamlines, bundle_name,
                                          metrics, labels,
                                          distance_values=None,
                                          correlation_values=None,
                                          density_weighting=False):
    """
    Compute the mean and std PER POiNT of the bundle for every given metric.

    Parameters
    ----------
    streamlines: list of numpy.ndarray
        Input streamlines under which to compute stats.
    bundle_name: str
        Name of the bundle. Will be used as a key in the dictionary.
    metrics: sequence
        list of nibabel objects representing the metrics files
    labels: np.ndarray
        List of labels obtained with scil_label_and_distance_maps.py
    distance_values: np.ndarray
        List of distances obtained with scil_compute_bundle_voxel_label_map.py
    correlation_values: np.ndarray
        List of correlations obtained with scil_compute_bundle_voxel_label_map.py
    density_weighting: bool
        If true, weight statistics by the number of streamlines passing through
        each voxel. [False]

    Returns
    -------
    stats
    """
    # Computing infos on bundle
    unique_labels = np.unique(labels)[1:]
    num_digits_labels = 3
    if density_weighting:
        streamline_count = compute_tract_counts_map(streamlines,
                                                    metrics[0].shape)
    else:
        streamline_count = np.ones(metrics[0].shape)
    streamline_count = streamline_count.astype(np.float64)

    # Bigger weight near the centroid streamline
    if isinstance(distance_values, np.ndarray):
        dist_to_centroid = 1.0 / distance_values
        dist_to_centroid[np.isinf(dist_to_centroid)] = -1
        dist_to_centroid[dist_to_centroid < 0] = np.max(dist_to_centroid)
    else:
        dist_to_centroid = 1

    # Get stats
    stats = {bundle_name: {}}
    for metric in metrics:
        metric_data = metric.get_fdata(dtype=np.float64)
        current_metric_fname, _ = split_name_with_nii(
            os.path.basename(metric.get_filename()))
        stats[bundle_name][current_metric_fname] = {}

        # Check if NaNs in metrics
        if np.any(np.isnan(metric_data)):
            logging.warning('Metric \"{}\" contains some NaN.'.format(metric.get_filename()) +
                            ' Ignoring voxels with NaN.')

        for i in unique_labels:
            number_key = '{}'.format(i).zfill(num_digits_labels)
            label_stats = {}
            stats[bundle_name][current_metric_fname][number_key] = label_stats

            label_metric = metric_data[labels == i]
            if density_weighting:
                label_weight = streamline_count[labels == i]
            else:
                label_weight = np.ones(label_metric.shape)

            if isinstance(distance_values, np.ndarray):
                label_weight *= dist_to_centroid[labels == i]
            if isinstance(correlation_values, np.ndarray):
                label_weight *= correlation_values[labels == i]
            if np.sum(label_weight) == 0:
                logging.warning('Weights sum to zero, can\'t be normalized. '
                                'Disabling weighting')
                label_weight = None

            # Check if NaNs in metrics
            label_masked_data = np.ma.masked_array(label_metric,
                                                   np.isnan(label_metric))
            label_mean = np.average(label_masked_data,
                                    weights=label_weight)
            label_std = np.sqrt(np.average(
                (label_masked_data - label_mean) ** 2,
                weights=label_weight))
            label_stats['mean'] = float(label_mean)
            label_stats['std'] = float(label_std)
    return stats


def plot_metrics_stats(means, stds, title=None, xlabel=None,
                       ylabel=None, figlabel=None, fill_color=None,
                       display_means=False):
    """
    Plots the mean of a metric along n points with the standard deviation.

    Parameters
    ----------
    means: Numpy 1D (or 2D) array of size n
        Mean of the metric along n points.
    stds: Numpy 1D (or 2D) array of size n
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
    display_means: bool
        Display the subjects means as semi-transparent line
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

    if means.ndim > 1:
        mean = np.average(means, axis=1)
        std = np.average(stds, axis=1)
        alpha = 0.5
    else:
        mean = np.array(means).ravel()
        std = np.array(stds).ravel()
        alpha = 0.9

    dim = np.arange(1, len(mean)+1, 1)

    if len(mean) <= 20:
        ax.xaxis.set_ticks(dim)

    ax.set_xlim(0, len(mean)+1)

    if means.ndim > 1 and display_means:
        for i in range(means.shape[-1]):
            ax.plot(dim, means[:, i], color="k", linewidth=1,
                    solid_capstyle='round', alpha=0.1)

    # Plot the mean line.
    ax.plot(dim, mean, color="k", linewidth=5, solid_capstyle='round')

    # Plot the std
    plt.fill_between(dim, mean - std, mean + std,
                     facecolor=fill_color, alpha=alpha)

    plt.close(fig)
    return fig
