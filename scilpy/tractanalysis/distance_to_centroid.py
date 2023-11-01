# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import KDTree


def transfer_and_diffuse_labels(target_sft, source_sft, nb_pts=20,):
    tree = KDTree(source_sft.streamlines._data, copy_data=True)
    pts_ids = tree.query_ball_point(target_sft.streamlines._data, r=4)

    max_count_labels = []
    for pts_id in pts_ids:
        if not pts_id:  # If no source point is close enough
            max_count_labels.append(-1)
            continue
        labels = np.mod(pts_id, nb_pts) + 1
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count_label = unique_labels[np.argmax(counts)]
        max_count_labels.append(max_count_label)
    
    labels = np.array(max_count_labels, dtype=np.uint16)

    curr_ind = 0
    for _, streamline in enumerate(target_sft.streamlines):
        next_ind = curr_ind + len(streamline)
        curr_labels = labels[curr_ind:next_ind]
        labels[curr_ind:next_ind] = diffuse_labels(streamline, curr_labels)
        curr_ind = next_ind

    return labels



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


def diffuse_labels(streamline, labels):
    """
    Replace -1 labels in the polyline using a diffusion algorithm.

    Parameters:
        streamline (ndarray): Coordinates of the polyline.
        labels (ndarray): Labels corresponding to the points in the polyline.

    Returns:
        ndarray: Updated labels with -1 replaced.
    """
    iteration = 0
    while np.any(labels == 65535):  # Continue until no -1 labels are left
        for i, label in enumerate(labels):
            if label == 65535:
                # Find closest point with a non-negative label
                min_distance = np.inf
                closest_label = -1
                for j, other_label in enumerate(labels):
                    if other_label != 65535:
                        distance = np.linalg.norm(streamline[i]-streamline[j])
                        if distance < min_distance:
                            min_distance = distance
                            closest_label = other_label
                # Update the label
                if iteration > 10:
                    labels[i] = 1
                labels[i] = closest_label
    return labels

from scipy.spatial.distance import pdist, squareform

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


def compute_shell_barycenters(labels_map):
    """
    Compute the barycenter for each label in a 3D NumPy array by maximizing
    the distance to the boundary.
    
    Parameters:
        labels_map (ndarray): The 3D array containing labels from 1-nb_pts.
        
    Returns:
        ndarray: An array of size (nb_pts, 3) containing the barycenter
        for each label.
    """
    labels = np.unique(labels_map)[1:]
    barycenters = np.zeros((len(labels), 3))

    for label in labels:
        mask = np.zeros_like(labels_map)
        mask[labels_map == label] = 1
        mask_coords = np.argwhere(mask)

        barycenter = find_medoid(mask_coords)
        barycenters[label - 1] = barycenter

    return barycenters


def compute_euclidean_barycenters(labels_map):
    """
    Compute the euclidean barycenter for each label in a 3D NumPy array.

    Parameters:
        labels_map (ndarray): The 3D array containing labels from 1-nb_pts.

    Returns:
        ndarray: A NumPy array of shape (nb_pts, 3) containing the barycenter
        for each label.
    """
    labels = np.unique(labels_map)[1:]
    barycenters = np.zeros((len(labels), 3))

    for label in labels:
        indices = np.argwhere(labels_map == label)
        if indices.size > 0:
            barycenter = np.mean(indices, axis=0)
            barycenters[label-1, :] = barycenter

    return barycenters
