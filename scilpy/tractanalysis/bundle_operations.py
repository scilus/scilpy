# -*- coding: utf-8 -*-

from itertools import count, takewhile
import logging

from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
import numpy as np
from dipy.tracking.streamlinespeed import set_number_of_points
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from scilpy.tractograms.streamline_and_mask_operations import \
    get_endpoints_density_map
from scilpy.tractograms.streamline_operations import \
    resample_streamlines_num_points, get_streamlines_bounding_box


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


def uniformize_bundle_sft(sft, axis=None, ref_bundle=None, swap=False):
    """Uniformize the streamlines in the given tractogram.

    Parameters
    ----------
    sft: StatefulTractogram
         The tractogram that contains the list of streamlines to be uniformized
    axis: int, optional
        Orient endpoints in the given axis
    ref_bundle: streamlines
        Orient endpoints the same way as this bundle (or centroid)
    swap: boolean, optional
        Swap the orientation of streamlines
    """
    old_space = sft.space
    old_origin = sft.origin
    sft.to_vox()
    sft.to_corner()
    density = get_endpoints_density_map(sft, point_to_select=3)
    indices = np.argwhere(density > 0)
    kmeans = KMeans(n_clusters=2, random_state=0, copy_x=True,
                    n_init=20).fit(indices)

    labels = np.zeros(density.shape)
    for i in range(len(kmeans.labels_)):
        labels[tuple(indices[i])] = kmeans.labels_[i]+1

    k_means_centers = kmeans.cluster_centers_
    main_dir_barycenter = np.argmax(
        np.abs(k_means_centers[0] - k_means_centers[-1]))

    if len(sft.streamlines) > 0:
        axis_name = ['x', 'y', 'z']
        if axis is None or ref_bundle is not None:
            if ref_bundle is not None:
                ref_bundle.to_vox()
                ref_bundle.to_corner()
                centroid = get_streamlines_centroid(ref_bundle.streamlines,
                                                    20)[0]
            else:
                centroid = get_streamlines_centroid(sft.streamlines, 20)[0]
            main_dir_ends = np.argmax(np.abs(centroid[0] - centroid[-1]))
            main_dir_displacement = np.argmax(
                np.abs(np.sum(np.gradient(centroid, axis=0), axis=0)))

            if main_dir_displacement != main_dir_ends \
                    or main_dir_displacement != main_dir_barycenter:
                logging.info('Ambiguity in orientation, you should use --axis')
            axis = axis_name[main_dir_displacement]
        logging.info('Orienting endpoints in the {} axis'.format(axis))
        axis_pos = axis_name.index(axis)

        if bool(k_means_centers[0][axis_pos] >
                k_means_centers[1][axis_pos]) ^ bool(swap):
            labels[labels == 1] = 3
            labels[labels == 2] = 1
            labels[labels == 3] = 2

        for i in range(len(sft.streamlines)):
            if ref_bundle:
                res_centroid = set_number_of_points(centroid, 20)
                res_streamlines = set_number_of_points(sft.streamlines[i], 20)
                norm_direct = np.sum(
                    np.linalg.norm(res_centroid - res_streamlines, axis=0))
                norm_flip = np.sum(
                    np.linalg.norm(res_centroid - res_streamlines[::-1],
                                   axis=0))
                if bool(norm_direct > norm_flip) ^ bool(swap):
                    sft.streamlines[i] = sft.streamlines[i][::-1]
                    for key in sft.data_per_point[i]:
                        sft.data_per_point[key][i] = \
                            sft.data_per_point[key][i][::-1]
            else:
                # Bitwise XOR
                if (bool(labels[tuple(sft.streamlines[i][0].astype(int))] >
                         labels[tuple(sft.streamlines[i][-1].astype(int))])
                        ^ bool(swap)):
                    sft.streamlines[i] = sft.streamlines[i][::-1]
                    for key in sft.data_per_point[i]:
                        sft.data_per_point[key][i] = \
                            sft.data_per_point[key][i][::-1]
    sft.to_space(old_space)
    sft.to_origin(old_origin)


def uniformize_bundle_sft_using_mask(sft, mask, swap=False):
    """Uniformize the streamlines in the given tractogram so head is closer to
    to a region of interest.

    Parameters
    ----------
    sft: StatefulTractogram
         The tractogram that contains the list of streamlines to be uniformized
    mask: np.ndarray
        Mask to use as a reference for the ROI.
    swap: boolean, optional
        Swap the orientation of streamlines
    """

    # barycenter = np.average(np.argwhere(mask), axis=0)
    old_space = sft.space
    old_origin = sft.origin
    sft.to_vox()
    sft.to_corner()

    tree = cKDTree(np.argwhere(mask))
    for i in range(len(sft.streamlines)):
        head_dist = tree.query(sft.streamlines[i][0])[0]
        tail_dist = tree.query(sft.streamlines[i][-1])[0]
        if bool(head_dist > tail_dist) ^ bool(swap):
            sft.streamlines[i] = sft.streamlines[i][::-1]
            for key in sft.data_per_point[i]:
                sft.data_per_point[key][i] = \
                    sft.data_per_point[key][i][::-1]

    sft.to_space(old_space)
    sft.to_origin(old_origin)


def detect_ushape(sft, minU, maxU):
    """
    Extract streamlines depending on their "u-shapeness".

    Parameters
    ----------
    sft: Statefull tractogram
        Tractogram used to extract streamlines depending on their ushapeness.
    minU: Float
        Minimum ufactor of a streamline.
    maxU: Float
        Maximum ufactor of a streamline.

    Returns
    -------
    list: the ids of u-shaped streamlines
        Only the ids are returned so proper filtering can be done afterwards.
    """
    ids = []
    new_sft = resample_streamlines_num_points(sft, 4)
    for i, s in enumerate(new_sft.streamlines):
        if len(s) == 4:
            first_point = s[0]
            last_point = s[-1]
            second_point = s[1]
            third_point = s[2]

            v1 = first_point - second_point
            v2 = second_point - third_point
            v3 = third_point - last_point

            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            v3 = v3 / np.linalg.norm(v3)

            val = np.dot(np.cross(v1, v2), np.cross(v2, v3))

            if minU <= val <= maxU:
                ids.append(i)

    return ids


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
                                                     nb_points=12,
                                                     min_threshold=0.5,
                                                     nb_samplings_max=30,
                                                     sampling_seed=1234,
                                                     fast_approx=False):
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
    resample_feature = ResampleFeature(nb_points=nb_points)
    metric = AveragePointwiseEuclideanMetric(resample_feature)

    box_min, box_max = get_streamlines_bounding_box(streamlines)

    # Half of the bounding box's halved diagonal length.
    initial_threshold = np.min(np.abs(box_max - box_min)) / 2.

    # Quickbundle's threshold is halved between hierarchical level.
    if fast_approx:
        thresholds = np.array([2 / 1.2**i for i in range(25)][1:])
        thresholds = np.concatenate(([40, 20, 10, 5, 2.5],
                                     thresholds[thresholds > min_threshold]))
    else:
        thresholds = takewhile(lambda t: t >= min_threshold,
                               (initial_threshold / 1.2**i for i in count()))
        thresholds = list(thresholds)

    ordering = np.arange(len(streamlines))
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

                for _, cluster in enumerate(clusters):
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


def remove_outliers(streamlines, threshold, nb_points=12, nb_samplings=30,
                    fast_approx=False):
    """
    Wrapper to classify inliers and outliers from a list of streamlines.

    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which inliers and outliers are separated.
    threshold: float
        Quickbundles distance threshold for the last threshold.
    nb_points: int
    nb_samplings: int
    fast_approx: bool

    Returns
    -------
    A tuple containing
        list: streamlines considered inliers
        list: streamlines considered outliers
    """
    summary = outliers_removal_using_hierarchical_quickbundles(
        streamlines, nb_points=nb_points, nb_samplings_max=nb_samplings,
        fast_approx=fast_approx)
    outliers_ids, inliers_ids = prune(streamlines, threshold, summary)

    return outliers_ids, inliers_ids

