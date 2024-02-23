# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import KDTree


def min_dist_to_centroid(bundle_pts, centroid_pts, nb_pts):
    """
    Compute minimal distance to centroids

    Parameters
    ----------
    bundles_pts: np.array
    centroid_pts: np.array
    nb_pts: int

    Returns
    -------
    Array:
    """
    tree = KDTree(centroid_pts, copy_data=True)
    dists, labels = tree.query(bundle_pts, k=1)
    dists, labels = np.expand_dims(
        dists, axis=1), np.expand_dims(labels, axis=1)

    labels = np.mod(labels, nb_pts)

    sum_dist = np.expand_dims(np.sum(dists, axis=1), axis=1)
    weights = np.exp(-dists / sum_dist)

    votes = []
    for i in range(len(bundle_pts)):
        vote = np.bincount(labels[i], weights=weights[i])
        total = np.arange(np.amax(labels[i])+1)
        winner = total[np.argmax(vote)]
        votes.append(winner)

    return np.array(votes, dtype=np.uint16), np.average(dists, axis=1)
