# -*- coding: utf-8 -*-
import copy

from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points
import numpy as np
from numpy.random import RandomState
from scipy.spatial import cKDTree
from sklearn.metrics import cohen_kappa_score

from scilpy.utils.streamlines import (perform_streamlines_operation,
                                      subtraction, intersection, union)


def get_endpoints_map(streamlines, dimensions, point_to_select=3):
    endpoints_map = np.zeros(dimensions)
    for streamline in streamlines:
        streamline = set_number_of_points(streamline, 99)
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
    if len(bundle_1) < 1 or len(bundle_2) < 1:
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


def binary_classification(segmentation_indices,
                          gold_standard_indices,
                          original_count,
                          mask_count=0):

    tp = len(np.intersect1d(segmentation_indices, gold_standard_indices))
    fp = len(np.setdiff1d(segmentation_indices, gold_standard_indices))
    fn = len(np.setdiff1d(gold_standard_indices, segmentation_indices))
    tn = len(np.setdiff1d(range(original_count),
                          np.union1d(segmentation_indices,
                                     gold_standard_indices)))
    if mask_count > 0:
        tn = tn - original_count + mask_count
    # Extreme that is not covered, all indices are in the gold standard
    # and the segmentation indices got them 100% right
    if tp == 0:
        sensitivity = np.nan
        specificity = np.nan
        precision = np.nan
        accuracy = np.nan
        dice = np.nan
        kappa = np.nan
        youden = np.nan
    else:
        sensitivity = tp / float(tp + fn)
        specificity = tn / float(tn + fp)
        precision = tp / float(tp + fp)
        accuracy = (tp + tn) / float(tp + fp + fn + tn)
        dice = 2 * tp / float(2 * tp + fp + fn)

        seg_arr = np.zeros((original_count,))
        gs_arr = np.zeros((original_count,))

        seg_arr[segmentation_indices] = 1
        gs_arr[gold_standard_indices] = 1

        # To make sure the amount of indices within the WM mask is consistent
        if mask_count > 0:
            empty_indices = np.where(seg_arr + gs_arr < 1)[0]
            indices_to_removes = original_count - mask_count
            seg_arr = np.delete(seg_arr, empty_indices[0:indices_to_removes])
            gs_arr = np.delete(gs_arr, empty_indices[0:indices_to_removes])

        kappa = cohen_kappa_score(seg_arr, gs_arr)
        youden = sensitivity + specificity - 1

    return sensitivity, specificity, precision, accuracy, dice, kappa, youden
