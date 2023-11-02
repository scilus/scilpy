# -*- coding: utf-8 -*-

import heapq

from dipy.tracking.metrics import length
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform


def min_dist_to_centroid(target_pts, source_pts, nb_pts=None,
                         pre_computed_labels=None):
    if nb_pts is None and pre_computed_labels is None:
        raise ValueError('Either nb_pts or labels must be provided.')

    tree = KDTree(source_pts, copy_data=True)
    _, labels = tree.query(target_pts, k=1)

    if pre_computed_labels is None:
        labels = np.mod(labels, nb_pts) + 1
    else:
        labels = pre_computed_labels[labels]

    return labels.astype(np.uint16)


def associate_labels(target_sft, source_sft,
                                nb_pts=20):
    kdtree = KDTree(source_sft.streamlines._data)

    # Initialize vote counters
    head_votes = np.zeros(nb_pts, dtype=int)
    tail_votes = np.zeros(nb_pts, dtype=int)

    for streamline in target_sft.streamlines:
        head = streamline[0]
        tail = streamline[-1]

        # Find closest IDs in the target
        closest_head_id = kdtree.query(head)[1]
        closest_tail_id = kdtree.query(tail)[1]

        # Knowing the centroids are already labels correctly, their
        # label is the modulo of the ID (based on nb_pts)
        closest_head_label = np.mod(closest_head_id, nb_pts) + 1
        closest_tail_label = np.mod(closest_tail_id, nb_pts) + 1
        head_votes[closest_head_label - 1] += 1
        tail_votes[closest_tail_label - 1] += 1

    # Trouver l'étiquette avec le plus de votes
    most_voted_head = np.argmax(head_votes) + 1
    most_voted_tail = np.argmax(tail_votes) + 1

    labels = []
    for i in range(len(target_sft)):
        streamline = target_sft.streamlines[i]
        lengths = np.insert(length(streamline, along=True), 0, 0)
        lengths = (lengths / np.max(lengths)) * \
            (most_voted_tail - most_voted_head) + most_voted_head

        labels = np.concatenate((labels, lengths))
        
    return labels.astype(np.uint16), most_voted_head, most_voted_tail


def find_medoid(points):
    """
    Find the medoid among a set of points.

    Parameters:
        points (ndarray): Points in N-dimensional space.

    Returns:
        ndarray: Coordinates of the medoid.
    """
    distance_matrix = squareform(pdist(points))
    medoid_idx = np.argmin(distance_matrix.sum(axis=1))
    return points[medoid_idx]


def compute_labels_map_barycenters(labels_map, euclidian=False, nb_pts=False):
    """
    Compute the barycenter for each label in a 3D NumPy array by maximizing
    the distance to the boundary.

    Parameters:
        labels_map (ndarray): The 3D array containing labels from 1-nb_pts.
        euclidian (bool): If True, the barycenter is the mean of the points

    Returns:
        ndarray: An array of size (nb_pts, 3) containing the barycenter
        for each label.
    """
    labels = np.arange(1, nb_pts+1) if nb_pts else np.unique(labels_map)[1:]
    barycenters = np.zeros((len(labels), 3))
    barycenters[:] = np.NAN

    for label in labels:
        indices = np.argwhere(labels_map == label)
        if indices.size > 0:
            mask = np.zeros_like(labels_map)
            mask[labels_map == label] = 1
            mask_coords = np.argwhere(mask)

            if euclidian:
                barycenter = np.mean(mask_coords, axis=0)
            else:
                barycenter = find_medoid(mask_coords)
            if labels_map[tuple(barycenter.astype(int))] != label:
                tree = KDTree(indices)
                _, ind = tree.query(barycenter, k=1)
                barycenter = indices[ind]

            barycenters[label - 1] = barycenter

    return np.array(barycenters)


def masked_manhattan_distance(mask, target_positions):
    """
    Compute the Manhattan distance from every position in a mask to a set of positions, 
    without stepping out of the mask.

    Parameters:
        mask (ndarray): A binary 3D array representing the mask.
        target_positions (list): A list of target positions within the mask.

    Returns:
        ndarray: A 3D array of the same shape as the mask, containing the Manhattan distances.
    """
    # Initialize distance array with infinite values
    distances = np.full(mask.shape, np.inf)

    # Initialize priority queue and set distance for target positions to zero
    priority_queue = []
    for x, y, z in target_positions:
        heapq.heappush(priority_queue, (0, (x, y, z)))
        distances[x, y, z] = 0

    # Directions for moving in the grid (Manhattan distance)
    directions = [(0, 0, 1), (0, 0, -1), (0, 1, 0),
                  (0, -1, 0), (1, 0, 0), (-1, 0, 0)]

    while priority_queue:
        current_distance, (x, y, z) = heapq.heappop(priority_queue)

        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz

            if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nz < mask.shape[2]:
                if mask[nx, ny, nz]:
                    new_distance = current_distance + 1

                    if new_distance < distances[nx, ny, nz]:
                        distances[nx, ny, nz] = new_distance
                        heapq.heappush(
                            priority_queue, (new_distance, (nx, ny, nz)))

    return distances
