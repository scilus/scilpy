# encoding: utf-8

import logging

from dipy.align.bundlemin import distance_matrix_mdf
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import set_number_of_points
import numpy as np


def remove_similar_streamlines(streamlines, threshold=5, do_avg=False):
    """ Remove similar streamlines, shuffling streamlines will impact the 
    results.
    Only provide a small set of streamlines (below 2000 if possible).

    Parameters
    ----------
    streamlines : list of numpy.ndarray
        Input streamlines to remove duplicates from.
    threshold : float
        Distance threshold to consider two streamlines similar, in mm.
    do_avg : bool
        Instead of removing similar streamlines, average all similar streamlines
        as a single smoother streamline.

    Returns
    -------
    streamlines : list of numpy.ndarray
    """
    if len(streamlines) == 1:
        return streamlines

    sample_20_streamlines = set_number_of_points(streamlines, 20)
    distance_matrix = distance_matrix_mdf(sample_20_streamlines,
                                          sample_20_streamlines)

    current_indice = 0
    avg_streamlines = []
    while True:
        sim_indices = np.where(distance_matrix[current_indice] < threshold)[0]

        pop_count = 0
        if do_avg:
            avg_streamline_list = []

        # Every streamlines similar to yourself (excluding yourself)
        # should be deleted from the set of desired streamlines
        for ind in sim_indices:
            if not current_indice == ind:
                streamlines.pop(ind-pop_count)

                distance_matrix = np.delete(distance_matrix, ind-pop_count,
                                            axis=0)
                distance_matrix = np.delete(distance_matrix, ind-pop_count,
                                            axis=1)
                pop_count += 1

            if do_avg:
                kicked_out = sample_20_streamlines[ind]
                avg_streamline_list.append(kicked_out)

        if do_avg:
            if len(avg_streamline_list) > 1:
                metric = AveragePointwiseEuclideanMetric()
                qb = QuickBundles(threshold=100, metric=metric)
                clusters = qb.cluster(avg_streamline_list)
                avg_streamlines.append(clusters.centroids[0])
            else:
                avg_streamlines.append(avg_streamline_list[0])

        current_indice += 1
        # Once you reach the end of the remaining streamlines
        if current_indice >= len(distance_matrix):
            break

    if do_avg:
        return avg_streamlines
    else:
        return streamlines


def subsample_clusters(cluster_map, streamlines, threshold,
                       min_cluster_size, average_streamlines=False):
    """ Using a cluster map, remove similar streamlines from all clusters
    independently using chunks of 1000 streamlines at the time to prevent
    infinite computation.

    Parameters
    ----------
    cluster_map : cluster_map class from QBx
        Contains the list of indices per cluster.
    Parameters
    ----------
    streamlines : list of numpy.ndarray
        Input streamlines to remove duplicates from.
    threshold : float
        Distance threshold to consider two streamlines similar, in mm.
    min_cluster_size : int
        Minimal cluster size to be considered. Clusters with less streamlines
        that the provided value will de discarded.
    average_streamlines : bool
        Instead of removing similar streamlines, average all similar streamlines
        as a single smoother streamline.

    Returns
    -------
    streamlines : list of numpy.ndarray
    """
    output_streamlines = []
    logging.debug('%s streamlines in tractogram', len(streamlines))
    logging.debug('%s clusters in tractogram', len(cluster_map))

    # Each cluster is processed independently
    for i in range(len(cluster_map)):
        if len(cluster_map[i].indices) < min_cluster_size:
            continue
        cluster_streamlines = [streamlines[j] for j in cluster_map[i].indices]
        size_before = len(cluster_streamlines)

        chunk_count = 0
        leftover_size = size_before
        cluster_sub_streamlines = []
        # Prevent matrix above 1M (n*n)
        while leftover_size > 0:
            start_id = chunk_count * 1000
            stop_id = (chunk_count + 1) * 1000
            partial_sub_streamlines = remove_similar_streamlines(
                cluster_streamlines[start_id:stop_id],
                threshold=threshold, do_avg=average_streamlines)

            # Add up the chunk results, update the loop values
            cluster_sub_streamlines.extend(partial_sub_streamlines)
            leftover_size -= 1000
            chunk_count += 1

        # Add up each cluster results
        output_streamlines.extend(cluster_sub_streamlines)

    return output_streamlines
