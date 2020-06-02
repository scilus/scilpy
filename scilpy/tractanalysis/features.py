# -*- coding: utf-8 -*-

from itertools import count, takewhile
import logging

from dipy.segment.clustering import QuickBundles, qbx_and_merge
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking import metrics as tm
import numpy as np


def remove_loops_and_sharp_turns(streamlines,
                                 max_angle,
                                 use_qb=False,
                                 qb_threshold=15.,
                                 qb_seed=0):
    """
    Remove loops and sharp turns from a list of streamlines.
    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which to remove loops and sharp turns.
    max_angle: float
        Maximal winding angle a streamline can have before
        being classified as a loop.
    use_qb: bool
        Set to True if the additional QuickBundles pass is done.
        This will help remove sharp turns. Should only be used on
        bundled streamlines, not on whole-brain tractograms.
    qb_threshold: float
        Quickbundles distance threshold, only used if use_qb is True.
    qb_seed: int
        Seed to initialize randomness in QuickBundles

    Returns
    -------
    list: the ids of clean streamlines
        Only the ids are returned so proper filtering can be done afterwards
    """

    streamlines_clean = []
    ids = []
    for i, s in enumerate(streamlines):
        if tm.winding(s) < max_angle:
            ids.append(i)
            streamlines_clean.append(s)

    if use_qb:
        if len(streamlines_clean) > 1:
            curvature = []

            rng = np.random.RandomState(qb_seed)
            clusters = qbx_and_merge(streamlines_clean,
                                     [40, 30, 20, qb_threshold],
                                     rng=rng, verbose=False)

            for cc in clusters.centroids:
                curvature.append(tm.mean_curvature(cc))
            mean_curvature = sum(curvature)/len(curvature)

            for i in range(len(clusters.centroids)):
                if tm.mean_curvature(clusters.centroids[i]) <= mean_curvature:
                    ids.extend(clusters[i].indices)
        else:
            logging.debug("Impossible to use the use_qb option because " +
                          "not more than one streamline left from the\n" +
                          "input file.")
    return ids


def get_streamlines_bounding_box(streamlines):
    """
    Classify inliers and outliers from a list of streamlines.
    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which inliers and outliers are separated.
    Returns
    -------
    tuple: Minimum and maximum corner coordinate of the streamlines
        bounding box
    """
    box_min = np.array([np.inf, np.inf, np.inf])
    box_max = -np.array([np.inf, np.inf, np.inf])

    for s in streamlines:
        box_min = np.minimum(box_min, np.min(s, axis=0))
        box_max = np.maximum(box_max, np.max(s, axis=0))

    return box_min, box_max


def prune(streamlines, threshold, features):
    """
    Discriminate streamlines based on a metrics, usually summary from function
    outliers_removal_using_hierarchical_quickbundles.
    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which inliers and outliers are separated.
    threshold: float
        Threshold use to discriminate streamlines using the feature.
    features: ndarray
        Values that represent a relevant metric to disciminate streamlines.
    Returns
    -------
    tuple:
        Indices for outliers (below threshold),
        indices for inliers (above threshold).
    """
    indices = np.arange(len(streamlines))

    outlier_indices = indices[features < threshold]
    rest_indices = indices[features >= threshold]

    return outlier_indices, rest_indices


def outliers_removal_using_hierarchical_quickbundles(streamlines,
                                                     min_threshold=0.5,
                                                     nb_samplings_max=30,
                                                     sampling_seed=1234):
    """
    Classify inliers and outliers from a list of streamlines.
    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which inliers and outliers are separated.
    min_threshold: float
        Quickbundles distance threshold for the last threshold.
    nb_samplings_max: int
        Number of run executed to explore the search space.
        A different sampling is used each time.
    sampling_seed: int
        Random number generation initialization seed.
    Returns
    -------
    ndarray: Float value representing the 0-1 score for each streamline
    """
    if nb_samplings_max < 2:
        raise ValueError("'nb_samplings_max' must be >= 2")

    rng = np.random.RandomState(sampling_seed)
    metric = "MDF_12points"

    box_min, box_max = get_streamlines_bounding_box(streamlines)

    # Half of the bounding box's halved diagonal length.
    initial_threshold = np.min(np.abs(box_max - box_min)) / 2.

    # Quickbundle's threshold is halved between hierarchical level.
    thresholds = list(takewhile(lambda t: t >= min_threshold,
                                (initial_threshold / 1.2**i for i in count())))

    ordering = np.arange(len(streamlines))
    nb_clusterings = 0
    path_lengths_per_streamline = 0

    streamlines_path = np.ones((len(streamlines), len(thresholds),
                                nb_samplings_max), dtype=int) * -1

    for i in range(nb_samplings_max):
        rng.shuffle(ordering)

        cluster_orderings = [ordering]
        for j, threshold in enumerate(thresholds):
            id_cluster = 0

            next_cluster_orderings = []
            qb = QuickBundles(metric=metric, threshold=threshold)
            for cluster_ordering in cluster_orderings:
                clusters = qb.cluster(streamlines, ordering=cluster_ordering)
                nb_clusterings += 1

                for k, cluster in enumerate(clusters):
                    streamlines_path[cluster.indices, j, i] = id_cluster
                    id_cluster += 1
                    if len(cluster) > 10:
                        next_cluster_orderings.append(cluster.indices)

            cluster_orderings = next_cluster_orderings

        if i <= 1:  # Needs at least two orderings to compute stderror.
            continue

        path_lengths_per_streamline = np.sum((streamlines_path != -1),
                                             axis=1)[:, :i]

    summary = np.mean(path_lengths_per_streamline,
                      axis=1) / np.max(path_lengths_per_streamline)
    return summary


def remove_outliers(streamlines, threshold):
    """
    Wrapper to classify inliers and outliers from a list of streamlines.
    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which inliers and outliers are separated.
    threshold: float
        Quickbundles distance threshold for the last threshold.
    -------
    A tuple containing
        list: streamlines considered inliers
        list: streamlines considered outliers
    """
    summary = outliers_removal_using_hierarchical_quickbundles(streamlines)
    outliers_ids, inliers_ids = prune(streamlines, threshold, summary)

    return outliers_ids, inliers_ids


def get_streamlines_centroid(streamlines, nb_points):
    """
    Compute centroid from streamlines using QuickBundles.

    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which we compute the centroid.
    nb_points: int
        Number of points defining the centroid streamline.

    Returns
    -------
    List of length one, containing a np.ndarray of shape (nb_points, 3)
    """
    resample_feature = ResampleFeature(nb_points=nb_points)
    quick_bundle = QuickBundles(
        threshold=np.inf,
        metric=AveragePointwiseEuclideanMetric(resample_feature))
    clusters = quick_bundle.cluster(streamlines)
    centroid_streamlines = clusters.centroids

    return centroid_streamlines
