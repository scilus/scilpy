# -*- coding: utf-8 -*-
import copy

from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points, length
import numpy as np
from numpy.random import RandomState
from scipy.spatial import cKDTree

from scilpy.utils.streamlines import (perform_streamlines_operation,
                                      subtraction, intersection, union)


def get_endpoints_map(streamlines, dimensions, point_to_select=3):
    """
    Compute an endpoints density map, supports selecting more than one points
    at each end.
    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines to compute endpoints density from.
    dimensions: tuple
        The shape of the reference volume for the streamlines.
    point_to_select: int
        Instead of computing the density based on the first and last points,
        select more than one at each end. To support compressed streamlines,
        a resampling to 0.5mm per segment is performed.
    Returns
    -------
    ndarray
        A ndarray where voxel values represent the density of endpoints.
    """
    endpoints_map = np.zeros(dimensions)
    for streamline in streamlines:
        streamline = set_number_of_points(streamline,
                                          int(length(streamline))*2)
        points_list = list(streamline[0:point_to_select, :].astype(int))
        points_list.extend(streamline[-(point_to_select+1):-1, :].astype(int))
        for xyz in points_list:
            x_val = int(np.clip(xyz[0], 0, dimensions[0]-1))
            y_val = int(np.clip(xyz[1], 0, dimensions[1]-1))
            z_val = int(np.clip(xyz[2], 0, dimensions[2]-1))
            endpoints_map[x_val, y_val, z_val] += 1

    return endpoints_map


def compute_bundle_adjacency_streamlines(bundle_1, bundle_2, non_overlap=False,
                                         centroids_1=None, centroids_2=None):
    """
    Compute the distance in millimeters between two bundles. Uses centroids
    to limit computation time. Each centroid of the first bundle is match
    to the nearest centroid of the second bundle and vice-versa. 
    Distance between matched paired is average for the final results.
    References
    ----------
    .. [Garyfallidis15] Garyfallidis et al. Robust and efficient linear 
        registration of white-matter fascicles in the space of streamlines, 
        Neuroimage, 2015.
    Parameters
    ----------
    bundle_1: list of ndarray
        First set of streamlines.
    bundle_2: list of ndarray
        Second set of streamlines.
    non_overlap: bool
        Exclude overlapping streamlines from the computation.
    centroids_1: list of ndarray
        Pre-computed centroids for the first bundle.
    centroids_2: list of ndarray
        Pre-computed centroids for the second bundle.
    Returns
    -------
    int
        Distance in millimeters between both bundles.
    """
    if not bundle_1 or not bundle_2:
        return -1
    thresholds = [32, 24, 12, 6]
    # Intialize the clusters
    if centroids_1 is None:
        centroids_1 = qbx_and_merge(bundle_1, thresholds, rng=RandomState(0),
                                    verbose=False).centroids
    if centroids_2 is None:
        centroids_2 = qbx_and_merge(bundle_2, thresholds, rng=RandomState(0),
                                    verbose=False).centroids
    if non_overlap:
        non_overlap_1, _ = perform_streamlines_operation(subtraction,
                                                         [bundle_1, bundle_2],
                                                         precision=0)
        non_overlap_2, _ = perform_streamlines_operation(subtraction,
                                                         [bundle_2, bundle_1],
                                                         precision=0)
        if non_overlap_1:
            non_overlap_centroids_1 = qbx_and_merge(non_overlap_1, thresholds,
                                                    rng=RandomState(0),
                                                    verbose=False).centroids
            distance_matrix_1 = bundles_distances_mdf(non_overlap_centroids_1,
                                                      centroids_2)

            min_b1 = np.min(distance_matrix_1, axis=0)
            distance_b1 = np.average(min_b1)
        else:
            distance_b1 = 0

        if non_overlap_2:
            non_overlap_centroids_2 = qbx_and_merge(non_overlap_2, thresholds,
                                                    rng=RandomState(0),
                                                    verbose=False).centroids
            distance_matrix_2 = bundles_distances_mdf(centroids_1,
                                                      non_overlap_centroids_2)
            min_b2 = np.min(distance_matrix_2, axis=1)
            distance_b2 = np.average(min_b2)
        else:
            distance_b2 = 0

    else:
        distance_matrix = bundles_distances_mdf(centroids_1, centroids_2)
        min_b1 = np.min(distance_matrix, axis=0)
        min_b2 = np.min(distance_matrix, axis=1)
        distance_b1 = np.average(min_b1)
        distance_b2 = np.average(min_b2)

    return (distance_b1 + distance_b2) / 2.0


def compute_bundle_adjacency_voxel(binary_1, binary_2, non_overlap=False):
    """
    Compute the distance in millimeters between two bundles in the voxel
    representation. Convert the bundles to binary masks. Each voxel of the 
    first bundle is match to the the nearest voxel of the second bundle and 
    vice-versa. 
    Distance between matched paired is average for the final results.
    Parameters
    ----------
    bundle_1: list of ndarray
        First set of streamlines.
    bundle_2: list of ndarray
        Second set of streamlines.
    non_overlap: bool
        Exclude overlapping streamlines from the computation.
    Returns
    -------
    int
        Distance in millimeters between both bundles.
    """
    b1_ind = np.argwhere(binary_1 > 0)
    b2_ind = np.argwhere(binary_2 > 0)
    b1_tree = cKDTree(b1_ind)
    b2_tree = cKDTree(b2_ind)

    distance_1, _ = b1_tree.query(b2_ind)
    distance_2, _ = b2_tree.query(b1_ind)

    if non_overlap:
        if not np.nonzero(distance_1)[0].size == 0:
            distance_b1 = np.mean(distance_1[np.nonzero(distance_1)])
        else:
            distance_b1 = 0

        if not np.nonzero(distance_2)[0].size == 0:
            distance_b2 = np.mean(distance_2[np.nonzero(distance_2)])
        else:
            distance_b2 = 0
    else:
        distance_b1 = np.mean(distance_1)
        distance_b2 = np.mean(distance_2)

    return (distance_b1 + distance_b2) / 2.0


def compute_dice_voxel(density_1, density_2):
    """
    Compute the overlap (dice coefficient) between two density maps (or binary).
    Parameters
    ----------
    density_1: ndarray
        Density (or binary) map computed from the first bundle
    density_1: ndarray of ndarray
        Density (or binary) map computed from the second bundle
    Returns
    -------
    float
        Value between 0 and 1 that represent the spatial aggrement between 
        both map.
    """
    binary_1 = copy.copy(density_1)
    binary_1[binary_1 > 0] = 1
    binary_2 = copy.copy(density_2)
    binary_2[binary_2 > 0] = 1

    numerator = 2 * np.count_nonzero(binary_1 * binary_2)
    denominator = np.count_nonzero(binary_1) + np.count_nonzero(binary_2)
    if denominator > 0:
        dice = numerator / float(denominator)
    else:
        dice = np.nan

    indices = np.nonzero(binary_1 * binary_2)
    overlap_1 = density_1[indices]
    overlap_2 = density_2[indices]
    w_dice = (np.sum(overlap_1) + np.sum(overlap_2))
    denominator = float(np.sum(density_1) + np.sum(density_2))
    if denominator > 0:
        w_dice /= denominator
    else:
        w_dice = np.nan

    return dice, w_dice


def compute_dice_streamlines(bundle_1, bundle_2):
    """
    Compute the overlap (dice coefficient) between two bundles.
    Both bundles need to come from the exact same tractogram.
    Parameters
    ----------
    bundle_1: list of ndarray
        First set of streamlines.
    bundle_2: list of ndarray
        Second set of streamlines.
    Returns
    -------
    float
        Value between 0 and 1 that represent the aggrement between both bundles.
    """
    streamlines_intersect, _ = perform_streamlines_operation(intersection,
                                                             [bundle_1, bundle_2],
                                                             precision=0)
    streamlines_union, _ = perform_streamlines_operation(union,
                                                         [bundle_1, bundle_2],
                                                         precision=0)

    numerator = 2 * len(streamlines_intersect)
    denominator = len(bundle_1) + len(bundle_2)
    if denominator > 0:
        dice = numerator / float(denominator)
    else:
        dice = np.nan

    return dice, streamlines_intersect, streamlines_union
