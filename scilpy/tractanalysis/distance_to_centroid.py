# -*- coding: utf-8 -*-

import heapq
import logging
import time

from dipy.tracking.metrics import length
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import scipy.ndimage as ndi
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.svm import SVC

from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractograms.streamline_operations import \
    resample_streamlines_num_points, resample_streamlines_step_size


def closest_match_to_centroid(bundle_pts, centroid_pts, nb_pts):
    """
    Assign a label to each point in the bundle_pts based on the closest
    centroid_pts. The labels are between 1 and nb_pts, where nb_pts is the
    number of points in the centroid_pts. The labels are assigned based on
    the order of the centroid_pts.

    The 3D points are expected to be in the same space.

    Typically the bundle_pts will be voxel indices (from argwhere) and the
    centroid_pts will be the 3D positions of a single streamline.

    Parameters
    ----------
    bundle_pts: np.array
        Coordinates of all streamlines (N x nb_pts x 3)
    centroid_pts: np.array
        Coordinates of all streamlines (nb_pts x 3)
    nb_pts: int
        Number of point for the association to centroids

    Returns
    -------
    Array:
        Labels (between 1 and nb_pts) for all bundle points
    """
    # A kdtree is used to find the nearest euclidian neighbors between
    # all voxels (bundle) and the streamline (centroid)
    tree = KDTree(centroid_pts, copy_data=True)
    _, labels = tree.query(bundle_pts, k=1)

    # No matter how many points are in the centroids, labels will be between
    # 1 and nb_pts
    labels = (labels / np.max(labels) * (nb_pts - 1)) + 1

    return labels.astype(np.uint16)


def associate_labels(target_sft, min_label=1, max_label=20):
    """
    Associate labels to the streamlines in a target SFT using their lengths.
    Even if unequal distance between points, the labels are interpolated
    linearly so all the points are labeled according to their position.

    min and max labels are used in case there is a cut in the bundle.

    Parameters:
    ----------
    target_sft: StatefulTractogram
        The target SFT to label, streamlines can be in any space.
    min_label: int
        Minimum label to use.
    max_label: int
        Maximum label to use.

    Returns:
    -------
    Array: np.uint16
        Labels for each point along the streamlines.
    """

    curr_ind = 0
    target_labels = np.zeros(target_sft.streamlines._data.shape[0],
                             dtype=float)
    for streamline in target_sft.streamlines:
        curr_length = np.insert(length(streamline, along=True), 0, 0)
        curr_labels = np.interp(curr_length,
                                [0, curr_length[-1]],
                                [min_label, max_label])
        curr_labels = np.round(curr_labels)
        target_labels[curr_ind:curr_ind+len(streamline)] = curr_labels
        curr_ind += len(streamline)

    return target_labels, target_sft.streamlines._data


def find_medoid(points, max_points=10000):
    """
    Find the medoid among a set of points. A medoid is a point that minimizes
    the sum of the distances to all other points. Unlike a barycenter, the
    medoid is guaranteed to be one of the points in the set.

    Parameters:
    ----------
    points: ndarray
        An array of 3D coordinates.
    max_points: int
        Maximum number of points to use for the computation (will randomly
        select points if the number of points is greater than max_points).

    Returns:
    -------
        np.array:
            The 3D coordinates of the medoid.
    """
    if len(points) > max_points:
        selected_indices = np.random.choice(len(points), max_points,
                                            replace=False)
        points = points[selected_indices]

    distance_matrix = squareform(pdist(points))
    medoid_idx = np.argmin(distance_matrix.sum(axis=1))
    return points[medoid_idx]


def compute_labels_map_barycenters(labels_map, is_euclidian=False,
                                   nb_pts=False):
    """
    Compute the barycenter for each label in a 3D NumPy array by maximizing
    the distance to the boundary.

    Parameters:
    ----------
    labels_map: (ndarray)
        The 3D array containing labels from 1-nb_pts.
        euclidian (bool): If True, the barycenter is the mean of the points
        in the mask. If False, the barycenter is the medoid of the points in
        the mask.
    is_euclidian: bool
        If True, the barycenter is the mean of the points in the mask.
        If False, the barycenter is the medoid of the points in the mask.
        This is useful for the hyperplane method.
    nb_pts: int
        Number of points to use for computing barycenters.

    Returns:
    -------
    ndarray:
        An array of size (nb_pts, 3) containing the barycenter
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
    ----------
    mask (ndarray):
        A binary 3D array representing the mask.
    target_positions (list):
        A list of target positions within the mask.

    Returns:
    -------
    ndarray:
        A 3D array of the same shape as the mask, containing the
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


def compute_distance_map(labels_map, binary_mask, nb_pts, use_manhattan=False):
    """
    Computes the distance map for each label in the labels_map.

    Parameters:
    ----------
    labels_map (numpy.ndarray):
        A 3D array representing the labels map.
    binary_mask (numpy.ndarray):
        A 3D binary map used to calculate barycenter binary map.
    nb_pts (int):
        Number of points to use for computing barycenters.
    use_manhattan (bool):
        If True, use the Manhattan distance instead of the Euclidian distance.

    Returns:
    -------
        numpy.ndarray: A 3D array representing the distance map.
    """
    barycenters = compute_labels_map_barycenters(labels_map,
                                                 is_euclidian=not use_manhattan,
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

    distance_map = np.zeros(binary_mask.shape, dtype=float)
    barycenter_strs = [barycenters[head-1:tail]]
    barycenter_bin = compute_tract_counts_map(barycenter_strs,
                                              binary_mask.shape)
    barycenter_bin[barycenter_bin > 0] = 1

    for label in range(head, tail+1):
        mask = np.zeros(labels_map.shape)
        mask[labels_map == label] = 1
        labels_coords = np.array(np.where(mask)).T
        if labels_coords.size == 0:
            continue

        barycenter_bin_intersect = barycenter_bin * mask
        barycenter_intersect_coords = np.array(
            np.nonzero(barycenter_bin_intersect), dtype=int).T

        if barycenter_intersect_coords.size == 0:
            continue

        if use_manhattan:
            coords = [tuple(coord) for coord in barycenter_intersect_coords]
            curr_dists = masked_manhattan_distance(binary_mask, coords)
            distance_map[labels_map == label] = \
                curr_dists[labels_map == label]
        else:
            distances = np.linalg.norm(
                barycenter_intersect_coords[:, np.newaxis] - labels_coords,
                axis=-1)
            distance_map[labels_map == label] = np.min(distances, axis=0)

    return distance_map


def correct_labels_jump(labels_map, streamlines, nb_pts):
    """
    Correct the labels jump in the labels map by cutting the streamlines
    where the jump is detected and keeping the longest chunk.

    This avoid loops in the labels map and ensure that the labels are
    consistent along the streamlines.

    Parameters:
    ----------
    labels_map (ndarray):
        A 3D array representing the labels map.
    streamlines (ArraySequence):
        The streamlines used to compute the labels map.
    nb_pts (int):
        Number of points to use for computing barycenters.

    Returns:
    -------
    ndarray: A 3D array representing the corrected labels map.
    """

    labels_data = ndi.map_coordinates(labels_map, streamlines._data.T - 0.5,
                                      order=0)
    binary_mask = np.zeros(labels_map.shape, dtype=np.uint8)
    binary_mask[labels_map > 0] = 1

    # It is not allowed that labels jumps labels for consistency
    # Streamlines should have continous labels
    final_streamlines = []
    final_labels = []
    curr_ind = 0
    for streamline in streamlines:
        next_ind = curr_ind + len(streamline)
        curr_labels = labels_data[curr_ind:next_ind].astype(int)
        curr_ind = next_ind

        # Flip streamlines so the labels increase (facilitate if/else)
        # Should always be ordered in nextflow pipeline
        gradient = np.ediff1d(curr_labels)

        is_flip = False
        if len(np.argwhere(gradient < 0)) > len(np.argwhere(gradient > 0)):
            streamline = streamline[::-1]
            curr_labels = curr_labels[::-1]
            gradient *= -1
            is_flip = True

        # Find jumps, cut them and find the longest
        max_jump = max(nb_pts // 5, 1)
        if len(np.argwhere(np.abs(gradient) > max_jump)) > 0:
            pos_jump = np.where(np.abs(gradient) > max_jump)[0] + 1
            split_chunk = np.split(curr_labels,
                                   pos_jump)
            # Find the longest chunk using a sort
            max_pos = np.argmax([len(chunk) for chunk in split_chunk])

            curr_labels = split_chunk[max_pos]
            streamline = np.split(streamline,
                                  pos_jump)[max_pos]
            gradient = np.ediff1d(curr_labels)

        if is_flip:
            streamline = streamline[::-1]
            curr_labels = curr_labels[::-1]
        final_streamlines.append(streamline)
        final_labels.append(curr_labels)

    # Once the streamlines abnormalities are corrected, we can
    # recompute the labels map with the new streamlines/labels
    final_labels = ArraySequence(final_labels)
    final_streamlines = ArraySequence(final_streamlines)

    modified_binary_mask = compute_tract_counts_map(final_streamlines,
                                                    binary_mask.shape)
    modified_binary_mask[modified_binary_mask > 0] = 1

    # Compute the KDTree for the new streamlines to find the closest
    # labels for each voxel
    kd_tree = KDTree(final_streamlines._data - 0.5)

    indices = np.array(np.nonzero(modified_binary_mask), dtype=int).T
    labels_map = np.zeros(labels_map.shape, dtype=np.uint16)

    # This correspond to 1 voxel distance (cross)
    neighbor_ids = kd_tree.query_ball_point(indices, r=1.0)

    # For each voxel, look at the labels of the neighbors and
    # assign the appropriate label
    # If the most frequent label is not > 25% of the neighbors, assign 0
    # If the max jump is too big, assign 0
    # If the neighbor is empty, assign 0
    # If the neighbor is 1, assign the label
    # If the neighbor is > 1, assign the most frequent label
    for ind, neighbor_id in zip(indices, neighbor_ids):
        if len(neighbor_id) == 0:
            continue
        elif len(neighbor_id) == 1:
            labels_map[tuple(ind)] = final_labels._data[neighbor_id]
            continue
        label_values = final_labels._data[neighbor_id]
        gradient = np.ediff1d(label_values)
        if np.max(gradient) > max_jump:
            continue
        else:
            unique, counts = np.unique(label_values, return_counts=True)
            max_count = np.argmax(counts)
            labels_map[tuple(ind)] = unique[max_count] \
                if counts[max_count] / sum(counts) > 0.25 else 0

    return labels_map * modified_binary_mask


def subdivide_bundles(sft, sft_centroid, binary_mask, nb_pts,
                      method='centerline', fix_jumps=True):
    """
    Function to divide a bundle into multiple section along its length.
    The resulting labels map is based on the binary_mask, but the streamlines
    are required for a few internal corrections (for consistency).

    The default is to use the euclidian/centerline method, which is fast and
    works well for most cases.

    The hyperplane method allows for more complex shapes and to split the bundles
    into subsections that follow the geometry of each kind of bundle.
    However, this method is slower and requires extra quality control to ensure
    that the labels are correct. This method requires a centroid file that
    contains multiple streamlines.

    Parameters:
    ----------
    sft (StatefulTractogram):
        Represent the streamlines to be subdivided, streamlines representation
        is useful fro the fix_jump parameter.
    sft_centroid (StatefulTractogram):
        Centroids used as a reference for subdivision.
    binary_mask (ndarray):
        Mask to be converted to a label mask
    nb_pts (int):
        Number of subdivision along streamlines' length
    method (str):
        Choice between centerline or hyperplane for subdivision
    fix_jumps (bool):
        Run the correction for streamlines to reduce big transition along
        its length.

    Returns:
    -------
    ndarray:
        A 3D array representing the labels map.
    """
    sft.to_vox()
    sft_centroid.to_vox()
    sft.to_corner()
    sft_centroid.to_corner()

    # This allows to have a more uniform (in size) first and last labels
    endpoints_extended = False
    if method == 'hyperplane' and nb_pts >= 5:
        nb_pts += 2
        endpoints_extended = True

    sft_centroid = resample_streamlines_num_points(sft_centroid, nb_pts)

    timer = time.time()

    indices = np.array(np.nonzero(binary_mask), dtype=int).T
    labels = closest_match_to_centroid(indices,
                                       sft_centroid[0].streamlines._data,
                                       nb_pts=nb_pts)
    logging.debug('Computed labels using the euclidian method '
                  f'in {round(time.time() - timer, 3)} seconds')

    if method == 'centerline':
        labels_map = np.zeros(binary_mask.shape, dtype=np.uint16)
        labels_map[tuple(indices.T)] = labels
    elif method == 'hyperplane':
        min_label, max_label = labels.min(), labels.max()
        del labels, indices
        logging.debug('Computing Labels using the hyperplane method.\n'
                      '\tThis can take a while...')
        # Select 2000 elements from the SFTs to train the classifier
        streamlines_length = [length(streamline)
                              for streamline in sft.streamlines]
        random_indices = np.random.choice(len(sft.streamlines), 2000)
        tmp_sft = resample_streamlines_step_size(
            sft[random_indices], np.min(streamlines_length) / nb_pts)

        # Associate the labels to the streamlines using the centroids as
        # reference (to handle shorter bundles due to missing data)
        mini_timer = time.time()
        labels, points = associate_labels(tmp_sft, min_label, max_label)

        kd_tree = KDTree(points)
        indices = np.array(np.nonzero(binary_mask), dtype=int).T

        nn_indices = kd_tree.query(indices, k=1)[1]
        labels, points = labels[nn_indices], points[nn_indices]

        logging.debug('\tAssociated labels to centroids in '
                      f'{round(time.time() - mini_timer, 3)} seconds')

        # Initialize the scaler
        mini_timer = time.time()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(points)
        scaled_streamline_data = scaler.transform(points)

        svc = SVC(C=1.0, kernel='rbf', random_state=1)

        svc.fit(X=scaled_streamline_data, y=labels)
        logging.debug('\tSVC fit of training data in '
                      f'{round(time.time() - mini_timer, 3)} seconds')

        # Scale the coordinates of the voxels
        # Skip every other voxel to speed up the process
        mini_timer = time.time()
        masked_binary_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        masked_binary_mask[::2, ::2, ::2] = binary_mask[::2, ::2, ::2]
        voxel_coords = np.array(np.where(masked_binary_mask)).T
        scaled_voxel_coords = scaler.transform(voxel_coords)

        # Predict the labels for the voxels
        labels = svc.predict(X=scaled_voxel_coords)
        logging.debug('\tSVC prediction of labels in '
                      f'{round(time.time() - mini_timer, 3)} seconds')

        labels_map = np.zeros(binary_mask.shape, dtype=np.uint16)
        labels_map[np.where(masked_binary_mask)] = labels

        # Find the missing labels and assign them to the nearest neighbor
        # using the KDTree (we skip every other voxel to speed up the process)
        missing_indices = np.argwhere(binary_mask - masked_binary_mask)
        valid_indices = np.argwhere(masked_binary_mask)

        kd_tree = KDTree(valid_indices)
        nn_indices = kd_tree.query(missing_indices, k=1)[1]
        labels_map[tuple(missing_indices.T)] = \
            labels_map[tuple(valid_indices[nn_indices].T)]

    if endpoints_extended:
        labels_map[labels_map == nb_pts] = nb_pts - 1
        labels_map[labels_map == 1] = 2
        labels_map[labels_map > 0] -= 1
        nb_pts -= 2

    logging.debug('Computed labels using the hyperplane method '
                  f'in {round(time.time() - timer, 3)} seconds')

    # Correct the labels jump to prevent discontinuities
    if fix_jumps:
        logging.debug('Correcting labels jump...')
        timer = time.time()
        tmp_sft = resample_streamlines_step_size(sft, 1.0)
        labels_map = correct_labels_jump(labels_map, tmp_sft.streamlines,
                                         nb_pts)
        logging.debug('Corrected labels jump in '
                      f'{round(time.time() - timer, 3)} seconds')

    return labels_map
