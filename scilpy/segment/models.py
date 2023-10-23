# encoding: utf-8

from dipy.align.bundlemin import distance_matrix_mdf
from dipy.tracking.streamline import set_number_of_points
import numpy as np


def remove_similar_streamlines(streamlines, threshold=5):
    """ Remove similar streamlines, shuffling streamlines will impact the
    results.
    Only provide a small set of streamlines (below 2000 if possible).

    Parameters
    ----------
    streamlines : list of numpy.ndarray
        Input streamlines to remove duplicates from.
    threshold : float
        Distance threshold to consider two streamlines similar, in mm.

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
    while True:
        sim_indices = np.where(distance_matrix[current_indice] < threshold)[0]

        pop_count = 0
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

        current_indice += 1
        # Once you reach the end of the remaining streamlines
        if current_indice >= len(distance_matrix):
            break

    return streamlines
