# -*- coding: utf-8 -*-
import heapq

from dipy.tracking.metrics import length
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform

from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


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
    source_kdtree = KDTree(source_sft.streamlines._data)
    final_labels = np.zeros(target_sft.streamlines._data.shape[0],
                            dtype=float)
    curr_ind = 0
    for streamline in target_sft.streamlines:
        distances, ids = source_kdtree.query(streamline,
                                             k=max(1, nb_pts // 5))

        valid_points = distances != np.inf

        curr_labels = np.mod(ids[valid_points], nb_pts) + 1

        head = np.min(curr_labels)
        tail = np.max(curr_labels)

        lengths = np.insert(length(streamline, along=True), 0, 0)[::-1]
        lengths = (lengths / np.max(lengths)) * \
            (head - tail) + tail

        final_labels[curr_ind:curr_ind+len(lengths)] = lengths
        curr_ind += len(lengths)

    return np.round(final_labels), head, tail


def find_medoid(points, max_points=10000):
    """
    Find the medoid among a set of points.

    Parameters:
        points (ndarray): Points in N-dimensional space.

    Returns:
        ndarray: Coordinates of the medoid.
    """
    if len(points) > max_points:
        selected_indices = np.random.choice(len(points), max_points,
                                            replace=False)
        points = points[selected_indices]

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
            # If the barycenter is not in the mask, find the closest point
            if labels_map[tuple(barycenter.astype(int))] != label:
                tree = KDTree(indices)
                _, ind = tree.query(barycenter, k=1)
                del tree
                barycenter = indices[ind]

            barycenters[label - 1] = barycenter

    return np.array(barycenters)


def masked_manhattan_distance(mask, target_positions):
    """
    Compute the Manhattan distance from every position in a mask to a set of
    positions, without stepping out of the mask.

    Parameters:
        mask (ndarray): A binary 3D array representing the mask.
        target_positions (list): A list of target positions within the mask.

    Returns:
        ndarray: A 3D array of the same shape as the mask, containing the
        Manhattan distances.
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

            if 0 <= nx < mask.shape[0] and \
                0 <= ny < mask.shape[1] and \
                    0 <= nz < mask.shape[2]:
                if mask[nx, ny, nz]:
                    new_distance = current_distance + 1

                    if new_distance < distances[nx, ny, nz]:
                        distances[nx, ny, nz] = new_distance
                        heapq.heappush(
                            priority_queue, (new_distance, (nx, ny, nz)))

    return distances


def compute_distance_map(labels_map, binary_map, is_euclidian, nb_pts):
    """
    Computes the distance map for each label in the labels_map.

    Parameters:
    labels_map (numpy.ndarray):
        A 3D array representing the labels map.
    binary_map (numpy.ndarray):
        A 3D binary map used to calculate barycenter binary map.
    hyperplane (bool):
        A flag to determine the type of distance calculation.
    nb_pts (int):
        Number of points to use for computing barycenters.

    Returns:
        numpy.ndarray: A 3D array representing the distance map.
    """
    barycenters = compute_labels_map_barycenters(labels_map,
                                                 is_euclidian=is_euclidian,
                                                 nb_pts=nb_pts)
    # If the first/last few points are NaN, remove them this indicates that the
    # head/tail are not 1-NB_PTS
    isnan = np.isnan(barycenters).all(axis=1)
    head = np.argmax(~isnan) + 1
    tail = len(isnan) - np.argmax(~isnan[::-1])

    # Identify the indices that do contain NaN values after/before head/tail
    tmp_barycenter = barycenters[head-1:tail]
    valid_indices = np.argwhere(
        ~np.isnan(tmp_barycenter).any(axis=1)).flatten()
    valid_data = tmp_barycenter[valid_indices]
    interpolated_data = np.array(
        [np.interp(np.arange(len(tmp_barycenter)),
                   valid_indices,
                   valid_data[:, i]) for i in range(tmp_barycenter.shape[1])]).T
    barycenters[head-1:tail] = interpolated_data

    distance_map = np.zeros(binary_map.shape, dtype=float)
    barycenter_strs = [barycenters[head-1:tail]]
    barycenter_bin = compute_tract_counts_map(barycenter_strs,
                                              binary_map.shape)
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

        if is_euclidian:
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


def correct_labels_jump(labels_map, streamlines, nb_pts):
    labels_data = ndi.map_coordinates(labels_map, streamlines._data.T - 0.5,
                                      order=0)

    # It is not allowed that labels jumps labels for consistency
    # Streamlines should have continous labels
    final_streamlines = []
    final_labels = []
    curr_ind = 0
    for streamline in streamlines:
        next_ind = curr_ind + len(streamline)
        curr_labels = labels_data[curr_ind:next_ind]
        curr_ind = next_ind

        # Flip streamlines so the labels increase (facilitate if/else)
        # Should always be ordered in nextflow pipeline
        gradient = np.gradient(curr_labels)
        if len(np.argwhere(gradient < 0)) > len(np.argwhere(gradient > 0)):
            streamline = streamline[::-1]
            curr_labels = curr_labels[::-1]

        # Find jumps, cut them and find the longest
        gradient = np.ediff1d(curr_labels)
        max_jump = max(nb_pts // 2, 1)
        if len(np.argwhere(np.abs(gradient) > max_jump)) > 0:
            pos_jump = np.where(np.abs(gradient) > max_jump)[0] + 1
            split_chunk = np.split(curr_labels,
                                   pos_jump)

            max_len = 0
            max_pos = 0
            for j, chunk in enumerate(split_chunk):
                if len(chunk) > max_len:
                    max_len = len(chunk)
                    max_pos = j

            curr_labels = split_chunk[max_pos]
            gradient_chunk = np.ediff1d(chunk)
            if len(np.unique(np.sign(gradient_chunk))) > 1:
                continue
            streamline = np.split(streamline,
                                  pos_jump)[max_pos]

        final_streamlines.append(streamline)
        final_labels.append(curr_labels)

    # Once the streamlines abnormalities are corrected, we can
    # recompute the labels map with the new streamlines/labels
    final_labels = ArraySequence(final_labels)
    final_streamlines = ArraySequence(final_streamlines)

    kd_tree = KDTree(final_streamlines._data)
    indices = np.array(np.nonzero(labels_map), dtype=int).T
    labels_map = np.zeros(labels_map.shape, dtype=np.uint16)

    for ind in indices:
        neighbor_dists, neighbor_ids = kd_tree.query(ind, k=5)

        if not len(neighbor_ids):
            continue

        labels_val = final_labels._data[neighbor_ids]
        sum_dists_vox = np.sum(neighbor_dists)
        weights = np.exp(-neighbor_dists / sum_dists_vox)

        vote = np.bincount(labels_val, weights=weights)
        total = np.arange(np.amax(labels_val+1))
        winner = total[np.argmax(vote)]
        labels_map[ind[0], ind[1], ind[2]] = winner

    return labels_map
