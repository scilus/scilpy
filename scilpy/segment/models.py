#!/usr/bin/env python
# encoding: utf-8

import logging

import numpy as np
from dipy.align.bundlemin import distance_matrix_mdf
from dipy.tracking.streamline import set_number_of_points



def remove_similar_streamlines(streamlines, threshold=5, do_avg=False):
    """ Remove similar streamlines, shuffle streamlines will impact the results.
    Only provide a small set of streamlines (below 2000 if possible).

    Parameters
    ----------
    streamlines : list of numpy.ndarray
    threshold : float
    streamlines : bool

    Returns
    -------
    streamlines : list of numpy.ndarray
    """
    if len(streamlines) == 1:
        return streamlines

    sample_15_streamlines = set_number_of_points(streamlines, 15)
    distance_matrix = distance_matrix_mdf(sample_15_streamlines,
                                          sample_15_streamlines)

    current_indice = 0
    avg_streamlines = []
    while True:
        sim_indices = np.where(distance_matrix[current_indice] < threshold)[0]

        pop_count = 0
        if do_avg:
            avg_streamline = sample_15_streamlines[current_indice]
        # Every streamlines similar to yourself (excluding yourself)
        # should be deleted from the set of desired streamlines
        for ind in sim_indices:
            if not current_indice == ind:
                kicked_out = streamlines.pop(ind-pop_count)
                if do_avg:
                    avg_streamline = add_streamline(kicked_out,
                                                    avg_streamline, 15)
                distance_matrix = np.delete(distance_matrix, ind-pop_count,
                                            axis=0)
                distance_matrix = np.delete(distance_matrix, ind-pop_count,
                                            axis=1)
                pop_count += 1
        if do_avg:
            avg_streamlines.append(avg_streamline)
        current_indice += 1
        # Once you reach the end of the remaining streamlines
        if current_indice >= len(distance_matrix):
            break

    if do_avg:
        return avg_streamlines
    else:
        return streamlines


def subsample_clusters(cluster_map, streamlines, min_distance,
                       min_cluster_size, average_streamlines=False):
    """ .

    Parameters
    ----------
    cluster_map : cluster_map class from QBx
    streamlines : list of numpy.ndarray
    min_distance : float
    min_cluster_size : int
    average_streamlines : bool

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
                threshold=min_distance, do_avg=average_streamlines)

            # Add up the chunk results, update the loop values
            cluster_sub_streamlines.extend(partial_sub_streamlines)
            leftover_size -= 1000
            chunk_count += 1

        # Add up each cluster results
        output_streamlines.extend(cluster_sub_streamlines)

    return output_streamlines
