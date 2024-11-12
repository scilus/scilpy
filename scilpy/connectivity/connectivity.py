# -*- coding: utf-8 -*-
import logging
import threading

from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np

d = threading.local()


def compute_triu_connectivity_from_labels(tractogram, data_labels,
                                          keep_background=False,
                                          hide_labels=None):
    """
    Compute a connectivity matrix.

    Parameters
    ----------
    tractogram: StatefulTractogram, or list[np.ndarray]
        Streamlines. A StatefulTractogram input is recommanded.
        When using directly with a list of streamlines, streamlinee must be in
        vox space, corner origin.
    data_labels: np.ndarray
        The loaded nifti image.
    keep_background: Bool
        By default, the background (label 0) is not included in the matrix.
        If True, label 0 is kept.
    hide_labels: Optional[List[int]]
        If not None, streamlines ending in a voxel with a given label are
        ignored (i.e. matrix is set to 0 for that label).

    Returns
    -------
    matrix: np.ndarray
        With use_scilpy: shape (nb_labels + 1, nb_labels + 1)
        Else, shape (nb_labels, nb_labels)
    ordered_labels: List
        The list of labels. Name of each row / column.
    start_labels: List
        For each streamline, the label at starting point.
    end_labels: List
        For each streamline, the label at ending point.
    """
    if isinstance(tractogram, StatefulTractogram):
        #  Vox space, corner origin
        # = we can get the nearest neighbor easily.
        # Coord 0 = voxel 0. Coord 0.9 = voxel 0. Coord 1 = voxel 1.
        tractogram.to_vox()
        tractogram.to_corner()
        streamlines = tractogram.streamlines
    else:
        streamlines = tractogram

    ordered_labels = list(np.sort(np.unique(data_labels)))
    assert ordered_labels[0] >= 0, "Only accepting positive labels."
    nb_labels = len(ordered_labels)
    logging.debug("Computing connectivity matrix for {} labels."
                  .format(nb_labels))

    matrix = np.zeros((nb_labels, nb_labels), dtype=int)
    start_labels = []
    end_labels = []

    for s in streamlines:
        start = ordered_labels.index(
            data_labels[tuple(np.floor(s[0, :]).astype(int))])
        end = ordered_labels.index(
            data_labels[tuple(np.floor(s[-1, :]).astype(int))])

        start_labels.append(start)
        end_labels.append(end)

        matrix[start, end] += 1
        if start != end:
            matrix[end, start] += 1

    matrix = np.triu(matrix)
    assert matrix.sum() == len(streamlines)

    # Rejecting background
    if not keep_background and ordered_labels[0] == 0:
        logging.debug("Rejecting background.")
        ordered_labels = ordered_labels[1:]
        matrix = matrix[1:, 1:]

    # Hiding labels
    if hide_labels is not None:
        for label in hide_labels:
            if label not in ordered_labels:
                logging.warning("Cannot hide label {} because it was not in "
                                "the data.".format(label))
                continue
            idx = ordered_labels.index(label)
            nb_hidden = np.sum(matrix[idx, :]) + np.sum(matrix[:, idx]) - \
                matrix[idx, idx]
            if nb_hidden > 0:
                logging.warning("{} streamlines had one or both endpoints "
                                "in hidden label {} (line/column {})"
                                .format(nb_hidden, label, idx))
                matrix[idx, :] = 0
                matrix[:, idx] = 0
            else:
                logging.info("No streamlines with endpoints in hidden label "
                             "{} (line/column {}) :)".format(label, idx))
            ordered_labels[idx] = ("Hidden label ({})".format(label))

    return matrix, ordered_labels, start_labels, end_labels
