# -*- coding: utf-8 -*-

from nibabel.streamlines.array_sequence import ArraySequence
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


def associate_labels(target_sft, source_sft, nb_pts=20):
    # KDTree for the target streamlines
    target_kdtree = KDTree(target_sft.streamlines._data)

    distances, _ = target_kdtree.query(source_sft.streamlines._data, k=1,
                                       distance_upper_bound=5)
    valid_points = distances != np.inf

    # Find the first and last indices of non-infinite values
    if valid_points.any():
        valid_points = np.mod(np.flatnonzero(valid_points), nb_pts)
        labels, count = np.unique(valid_points, return_counts=True)
        count = count / np.sum(count)
        count[count < 1.0 / (nb_pts*1.5)] = np.NaN
        valid_indices = np.where(~np.isnan(count))[0]

        # Find the first and last non-NaN indices
        head = labels[valid_indices[0]] + 1
        tail = labels[valid_indices[-1]] + 1

    labels = []
    for i in range(len(target_sft)):
        streamline = target_sft.streamlines[i]
        lengths = np.insert(length(streamline, along=True), 0, 0)[::-1]
        lengths = (lengths / np.max(lengths)) * \
            (head - tail) + tail

        labels = np.concatenate((labels, lengths))

    return np.round(labels), head, tail


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


def compute_labels_map_barycenters(labels_map, is_euclidian=False, nb_pts=False):
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

            if is_euclidian:
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


import numpy as np
import scipy.ndimage as ndi
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
def compute_distance_map(labels_map, binary_map, new_labelling, nb_pts):
    """
    Computes the distance map for each label in the labels_map.

    Parameters:
    labels_map (numpy.ndarray): A 3D array representing the labels map.
    binary_map (numpy.ndarray): A 3D binary map used to calculate barycenter binary map.
    new_labelling (bool): A flag to determine the type of distance calculation.
    nb_pts (int): Number of points to use for computing barycenters.

    Returns:
    numpy.ndarray: A 3D array representing the distance map.
    """
    barycenters = compute_labels_map_barycenters(labels_map,
                                                 is_euclidian=new_labelling,
                                                 nb_pts=nb_pts)

    isnan = np.isnan(barycenters).all(axis=1)
    head = np.argmax(~isnan) + 1
    tail = len(isnan) - np.argmax(~isnan[::-1])

    distance_map = np.zeros(binary_map.shape, dtype=float)
    barycenter_strs = [barycenters[head-1:tail]]
    barycenter_bin = compute_tract_counts_map(barycenter_strs, binary_map.shape)
    barycenter_bin[barycenter_bin > 0] = 1

    for label in range(head, tail+1):
        mask = np.zeros(labels_map.shape)
        mask[labels_map == label] = 1
        labels_coords = np.array(np.where(mask)).T
        if labels_coords.size == 0:
            continue

        barycenter_bin_intersect = barycenter_bin * mask
        barycenter_intersect_coords = np.array(np.nonzero(barycenter_bin_intersect),
                                               dtype=int).T

        if barycenter_intersect_coords.size == 0:
            continue

        if not new_labelling:
            distances = np.linalg.norm(
                barycenter_intersect_coords[:, np.newaxis] - labels_coords,
                axis=-1)
            distance_map[labels_map == label] = np.min(distances, axis=0)
        else:
            coords = [tuple(coord) for coord in barycenter_intersect_coords]
            curr_dists = masked_manhattan_distance(binary_map, coords)
            distance_map[labels_map == label] = \
                curr_dists[labels_map == label]
    
    return distance_map
