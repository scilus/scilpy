# -*- coding: utf-8 -*-


from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamline import length, set_number_of_points
import numpy as np
from numpy.random import RandomState
from scipy.spatial import cKDTree
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import KDTree

from scilpy.utils.streamlines import (difference_robust,
                                      intersection_robust,
                                      union_robust)


def binary_classification(segmentation_indices,
                          gold_standard_indices,
                          original_count,
                          mask_count=0):
    """
    Compute all the binary classification measures using only indices from
    a dataset and its ground truth in any representation (voxels
    or streamlines).
    ----------
    segmentation_indices: list of int
        Indices of the data that are part of the segmentation.
    gold_standard_indices: list of int
        Indices of the ground truth.
    original_count: int
        Total size of the original dataset (before segmentation),
        e.g len(streamlines) or np.prod(mask.shape).
    mask_count: int
        Number of non-zeros voxels in the original dataset.
        Needed in order to get a valid true positive count for the voxel
        representation.
    Returns
    -------
    A tuple containing
        float: Value between 0 and 1 that represent the spatial aggrement
            between both bundles.
        list of ndarray: intersection_robust of streamlines in both bundle
        list of ndarray: union_robust of streamlines in both bundle
    """
    tp = len(np.intersect1d(segmentation_indices, gold_standard_indices))
    fp = len(np.setdiff1d(segmentation_indices, gold_standard_indices))
    fn = len(np.setdiff1d(gold_standard_indices, segmentation_indices))
    tn = len(np.setdiff1d(
        range(original_count), np.union1d(
            segmentation_indices, gold_standard_indices)))
    if mask_count > 0:
        tn = tn - original_count + mask_count
    # Extreme that is not covered, all indices are in the gold standard
    # and the segmentation indices got them 100% right
    if tp == 0:
        sensitivity = 0
        specificity = 0
        precision = 0
        accuracy = 0
        dice = 0
        kappa = 0
        youden = -1
    else:
        sensitivity = tp / float(tp + fn)
        if np.isclose(tn + fp, 0):
            specificity = 1
        else:
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


def get_endpoints_density_map(streamlines, dimensions, point_to_select=1):
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
    ndarray: A ndarray where voxel values represent the density of endpoints.
    """
    endpoints_map_head, endpoints_map_tail = \
        get_head_tail_density_maps(streamlines, dimensions, point_to_select)
    return endpoints_map_head + endpoints_map_tail


def get_head_tail_density_maps(streamlines, dimensions, point_to_select=1):
    """
    Compute two separate endpoints density maps for the head and tail
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
    A tuple containing
        ndarray: A ndarray where voxel values represent the density of
            head endpoints.
        ndarray: A ndarray where voxel values represent the density of
            tail endpoints.
    """
    endpoints_map_head = np.zeros(dimensions)
    endpoints_map_tail = np.zeros(dimensions)
    for streamline in streamlines:
        nb_point = max(2,  int(length(streamline))*2)
        streamline = set_number_of_points(streamline,
                                          nb_point)
        points_list_head = \
            list(streamline[0:point_to_select, :])
        points_list_tail = \
            list(streamline[-point_to_select:, :])
        for xyz in points_list_head:
            x_val = np.clip(xyz[0], 0, dimensions[0]-1).astype(int)
            y_val = np.clip(xyz[1], 0, dimensions[1]-1).astype(int)
            z_val = np.clip(xyz[2], 0, dimensions[2]-1).astype(int)
            endpoints_map_head[x_val, y_val, z_val] += 1
        for xyz in points_list_tail:
            x_val = np.clip(xyz[0], 0, dimensions[0]-1).astype(int)
            y_val = np.clip(xyz[1], 0, dimensions[1]-1).astype(int)
            z_val = np.clip(xyz[2], 0, dimensions[2]-1).astype(int)
            endpoints_map_tail[x_val, y_val, z_val] += 1
    return endpoints_map_head, endpoints_map_tail


def approximate_surface_node(roi):
    """
    Compute the number of surface voxels (i.e. nodes connected to at least one
    zero-valued neighboring voxel)
    Parameters
    ----------
    roi: ndarray
        A ndarray where voxel values represent the density of a bundle and
        it is binarized.

    Returns
    -------
    int: the number of surface voxels
    """
    ind = np.argwhere(roi > 0)
    tree = KDTree(ind)
    count = np.sum(7 - tree.query_radius(ind, r=1.0,
                                         count_only=True))

    return count


def compute_fractal_dimension(density, n_steps=10, box_size_min=1.0,
                              box_size_max=2.0, threshold=0.0, box_size=None):
    """
    Compute the fractal dimension of a bundle to measure the roughness.
    The code is extracted from https://github.com/FBK-NILab/fractal_dimension
    Parameters. The result is dependent on voxel size and the number of voxels.
    If data comparison is performed, the bundles MUST be in same resolution.
    ----------
    density: ndarray
        A ndarray where voxel values represent the density of a bundle. This
        function computes the fractal dimension of the bundle.
    n_steps: int
        The number of box sizes used to approximate fractal dimension. A larger
        number of steps will increase the accuracy of the approximation, but
        will also take more time. The default number of boxes sizes is 10.
    box_size_min: float
        The minimum size of boxes. This number should be larger than or equal
        to 1.0 and is defaulted to 1.0.
    box_size_max: float
        The maximum size of boxes. This number should be larger than the
        minimum size of boxes.
    threshold: float
        The threshold to filter the voxels in the density array. The default is
        set to 0, so only nonzero voxels will be considered.
    box_size: ndarray
        A ndarray of different sizes of boxes in a linear space in an ascending
        order.
    Returns
    -------
    float: fractal dimension of a bundle
    """
    pixels = np.array(np.where(density > threshold)).T

    if box_size is None:
        box_size = np.linspace(box_size_min, box_size_max, n_steps)

    counts = np.zeros(len(box_size))
    for i, bs in enumerate(box_size):
        bins = \
            [np.arange(0, image_side + bs, bs) for image_side in density.shape]
        H, edges = np.histogramdd(pixels, bins=bins)
        counts[i] = (H > 0).sum()

    if (counts < 1).any():
        fractal_dimension = 0.0
    else:
        # linear regression:
        coefficients = np.polyfit(np.log(box_size), np.log(counts), 1)
        fractal_dimension = -coefficients[0]

    return fractal_dimension


def compute_bundle_adjacency_streamlines(bundle_1, bundle_2, non_overlap=False,
                                         centroids_1=None, centroids_2=None):
    """
    Compute the distance in millimeters between two bundles. Uses centroids
    to limit computation time. Each centroid of the first bundle is matched
    to the nearest centroid of the second bundle and vice-versa.
    Distance between matched paired is averaged for the final results.
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
    float: Distance in millimeters between both bundles.
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
        non_overlap_1, _ = difference_robust([bundle_1, bundle_2])
        non_overlap_2, _ = difference_robust([bundle_2, bundle_1])

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
    first bundle is matched to the the nearest voxel of the second bundle and
    vice-versa.
    Distance between matched paired is averaged for the final results.
    Parameters
    ----------
    binary_1: ndarray
        Binary mask computed from the first bundle
    binary_2: ndarray
        Binary mask computed from the second bundle
    non_overlap: bool
        Exclude overlapping voxels from the computation.
    Returns
    -------
    float: Distance in millimeters between both bundles.
    """
    b1_ind = np.argwhere(binary_1 > 0)
    b2_ind = np.argwhere(binary_2 > 0)
    b1_tree = cKDTree(b1_ind)
    b2_tree = cKDTree(b2_ind)

    distance_1, _ = b1_tree.query(b2_ind)
    distance_2, _ = b2_tree.query(b1_ind)

    if non_overlap:
        non_zeros_1 = np.nonzero(distance_1)
        non_zeros_2 = np.nonzero(distance_2)
        if not non_zeros_1[0].size == 0:
            distance_b1 = np.mean(distance_1[non_zeros_1])
        else:
            distance_b1 = 0

        if not non_zeros_2[0].size == 0:
            distance_b2 = np.mean(distance_2[non_zeros_2])
        else:
            distance_b2 = 0
    else:
        distance_b1 = np.mean(distance_1)
        distance_b2 = np.mean(distance_2)

    return (distance_b1 + distance_b2) / 2.0


def compute_dice_voxel(density_1, density_2):
    """
    Compute the overlap (dice coefficient) between two
    density maps (or binary).
    Parameters
    ----------
    density_1: ndarray
        Density (or binary) map computed from the first bundle
    density_2: ndarray
        Density (or binary) map computed from the second bundle
    Returns
    -------
    A tuple containing
        float: Value between 0 and 1 that represent the spatial aggrement
            between both bundles.
        float: Value between 0 and 1 that represent the spatial aggrement
            between both bundles, weighted by streamlines density.
    """
    overlap_idx = np.nonzero(density_1 * density_2)
    numerator = 2 * len(overlap_idx[0])
    denominator = np.count_nonzero(density_1) + np.count_nonzero(density_2)

    if denominator > 0:
        dice = numerator / float(denominator)
    else:
        dice = np.nan

    overlap_1 = density_1[overlap_idx]
    overlap_2 = density_2[overlap_idx]
    w_dice = np.sum(overlap_1) + np.sum(overlap_2)
    denominator = np.sum(density_1) + np.sum(density_2)
    if denominator > 0:
        w_dice /= denominator
    else:
        w_dice = np.nan

    return dice, w_dice


def compute_correlation(density_1, density_2):
    """
    Compute the overlap (dice coefficient) between two density
    maps (or binary). Correlation being less robust to extreme
    case (no overlap, identical array), a lot of check a needed to prevent NaN.
    Parameters
    ----------
    density_1: ndarray
        Density (or binary) map computed from the first bundle
    density_2: ndarray
        Density (or binary) map computed from the second bundle
    Returns
    -------
    float: Value between 0 and 1 that represent the spatial aggrement
        between both bundles taking into account density.
    """
    indices = np.where(density_1 + density_2 > 0)
    if np.array_equal(density_1, density_2):
        density_correlation = 1
    elif (np.sum(density_1) > 0 and np.sum(density_2) > 0) \
            and np.count_nonzero(density_1 * density_2):
        density_correlation = np.corrcoef(density_1[indices],
                                          density_2[indices])[0, 1]
    else:
        density_correlation = 0

    return max(0, density_correlation)


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
    A tuple containing
        float: Value between 0 and 1 that represent the spatial aggrement
            between both bundles.
        list of ndarray: intersection_robust of streamlines in both bundle
        list of ndarray: union_robust of streamlines in both bundle
    """
    streamlines_intersect, _ = intersection_robust([bundle_1, bundle_2])
    streamlines_union_robust, _ = union_robust([bundle_1, bundle_2])

    numerator = 2 * len(streamlines_intersect)
    denominator = len(bundle_1) + len(bundle_2)
    if denominator > 0:
        dice = numerator / float(denominator)
    else:
        dice = np.nan

    return dice, streamlines_intersect, streamlines_union_robust
