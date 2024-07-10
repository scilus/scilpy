# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import logging
import warnings

from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.segment.clustering import qbx_and_merge
from dipy.segment.fss import FastStreamlineSearch
from dipy.tracking.distances import bundles_distances_mdf
import numpy as np
from numpy.random import RandomState
from scipy.spatial import cKDTree
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import KDTree
from tqdm import tqdm

from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.todi import TrackOrientationDensityImaging
from scilpy.tractograms.streamline_operations import generate_matched_points
from scilpy.tractograms.tractogram_operations import (difference_robust,
                                                      intersection_robust,
                                                      union_robust)
from scilpy.image.volume_operations import (normalize_metric, merge_metrics)
from scilpy.image.volume_math import neighborhood_correlation_


def binary_classification(segmentation_indices,
                          gold_standard_indices,
                          original_count,
                          mask_count=0):
    """
    Compute all the binary classification measures using only indices from
    a dataset and its ground truth in any representation (voxels
    or streamlines).

    Parameters
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

    - float: Value between 0 and 1 that represent the spatial aggrement
        between both bundles.
    - list of ndarray: intersection_robust of streamlines in both bundle
    - list of ndarray: union_robust of streamlines in both bundle
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

    Parameters
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
    A tuple containing:

    - float: Value between 0 and 1 that represent the spatial aggrement
        between both bundles.
    - float: Value between 0 and 1 that represent the spatial aggrement
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
    - float: Value between 0 and 1 that represent the spatial aggrement
        between both bundles.
    - list of ndarray: intersection_robust of streamlines in both bundle
    - list of ndarray: union_robust of streamlines in both bundle
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


def _compute_difference_for_voxel(chunk_indices,
                                  skip_streamlines_distance=False):
    """
    Compute the difference between two sets of streamlines for a given voxel.
    This function uses global variable to avoid duplicating the data for each
    chunk of voxels.

    Use the function tractogram_pairwise_comparison() as an entry point.
    To differentiate empty voxels from voxels with no data, the function
    returns NaN if no data is found.

    Parameters
    ----------
    chunk_indices: list
        List of indices of the voxel to process.
    skip_streamlines_distance: bool
        If true, skip the computation of the distance between streamlines.

    Returns
    -------
    results: list
        List of the computed differences in the same order as the input voxels.
    """
    global sft_1, sft_2, matched_points_1, matched_points_2, tree_1, tree_2, \
        sh_data_1, sh_data_2
    results = []
    for vox_ind in chunk_indices:
        vox_ind = tuple(vox_ind)

        global B
        has_data = sh_data_1[vox_ind].any() and sh_data_2[vox_ind].any()
        if has_data:
            sf_1 = np.dot(sh_data_1[vox_ind], B)
            sf_2 = np.dot(sh_data_2[vox_ind], B)
            acc = np.corrcoef(sf_1, sf_2)[0, 1]
        else:
            acc = np.nan

        if skip_streamlines_distance:
            results.append([np.nan, acc])
            continue

        # Get the streamlines in the first SFT neighborhood (i.e., 1.5mm away)
        pts_ind_1 = tree_1.query_ball_point(vox_ind, 1.5)
        if not pts_ind_1:
            results.append([np.nan, acc])
            continue
        strs_ind_1 = np.unique(matched_points_1[pts_ind_1])
        neighb_streamlines_1 = sft_1.streamlines[strs_ind_1]

        # Get the streamlines in the second SFT neighborhood (i.e., 1.5mm away)
        pts_ind_2 = tree_2.query_ball_point(vox_ind, 1.5)
        if not pts_ind_2:
            results.append([np.nan, acc])
            continue
        strs_ind_2 = np.unique(matched_points_2[pts_ind_2])
        neighb_streamlines_2 = sft_2.streamlines[strs_ind_2]

        # Using neighb_streamlines (all streamlines in the neighborhood of our
        # voxel), we can compute the distance between the two sets of
        # streamlines using FSS (FastStreamlineSearch).
        with warnings.catch_warnings(record=True) as _:
            fss = FastStreamlineSearch(neighb_streamlines_1, 10, resampling=12)
            dist_mat = fss.radius_search(neighb_streamlines_2, 10)
            sparse_dist_mat = np.abs(dist_mat.tocsr()).toarray()
            sparse_ma_dist_mat = np.ma.masked_where(sparse_dist_mat < 1e-3,
                                                    sparse_dist_mat)
            sparse_ma_dist_vec = np.squeeze(np.min(sparse_ma_dist_mat,
                                                   axis=0))

            # dists will represent the average distance between the two sets of
            # streamlines in the neighborhood of the voxel.
            dist = np.average(sparse_ma_dist_vec)
            results.append([dist, acc])

    return results


def _compare_tractogram_wrapper(mask, nbr_cpu, skip_streamlines_distance):
    """
    Wrapper for the comparison of two tractograms. This function uses
    multiprocessing to compute the difference between two sets of streamlines
    for each voxel.

    This function simply calls the function _compute_difference_for_voxel(),
    which expect chunks of indices to process and use global variables to avoid
    duplicating the data for each chunk of voxels.

    Use the function tractogram_pairwise_comparison() as an entry point.

    Parameters
    ----------
    mask: np.ndarray
        Mask of the data to compare.
    nbr_cpu: int
        Number of CPU to use.
    skip_streamlines_distance: bool
        If true, skip the computation of the distance between streamlines.

    Returns
    -------
    diff_data: np.ndarray
        Array containing the computed differences (mm).
    acc_data: np.ndarray
        Array containing the computed angular correlation.
    """
    dimensions = mask.shape

    # Initialize multiprocessing
    indices = np.argwhere(mask > 0)
    diff_data = np.zeros(dimensions)
    diff_data[:] = np.nan
    acc_data = np.zeros(dimensions)
    acc_data[:] = np.nan

    def chunked_indices(indices, chunk_size=1000):
        """Yield successive chunk_size chunks from indices."""
        for i in range(0, len(indices), chunk_size):
            yield indices[i:i + chunk_size]

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=len(indices))

    # Create chunks of indices
    np.random.shuffle(indices)
    index_chunks = list(chunked_indices(indices))

    with ProcessPoolExecutor(max_workers=nbr_cpu) as executor:
        futures = {executor.submit(
            _compute_difference_for_voxel, chunk,
            skip_streamlines_distance): chunk for chunk in index_chunks}

        for future in as_completed(futures):
            chunk = futures[future]
            try:
                results = future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')
            else:
                results = np.array(results)
                diff_data[tuple(chunk.T)] = results[:, 0]
                acc_data[tuple(chunk.T)] = results[:, 1]

            # Update tqdm progress bar
            progress_bar.update(len(chunk))

    return diff_data, acc_data


def tractogram_pairwise_comparison(sft_one, sft_two, mask, nbr_cpu=1,
                                   skip_streamlines_distance=True):
    """
    Compute the difference between two sets of streamlines for each voxel in
    the mask. This function uses multiprocessing to compute the difference
    between two sets of streamlines for each voxel.

    Parameters
    ----------
    sft_one: StatefulTractogram
        First tractogram to compare.
    sft_two: StatefulTractogram
        Second tractogram to compare.
    mask: np.ndarray
        Mask of the data to compare (optional).
    nbr_cpu: int
        Number of CPU to use (default: 1).
    skip_streamlines_distance: bool
        If true, skip the computation of the distance between streamlines.
        (default: True)

    Returns
    -------
    acc_norm: np.ndarray
        Angular correlation coefficient.
    corr_norm: np.ndarray
        Correlation coefficient of density maps.
    diff_norm: np.ndarray
        Voxelwise distance between sets of streamlines.
    heatmap: np.ndarray
        Merged heatmap of the three metrics using harmonic mean.
    mask: np.ndarray
        Final mask. Intersection of given mask (if any) and density masks of
        both tractograms.
    """
    global sft_1, sft_2
    sft_1, sft_2 = sft_one, sft_two

    sft_1.to_vox()
    sft_2.to_vox()
    sft_1.streamlines._data = sft_1.streamlines._data.astype(np.float16)
    sft_2.streamlines._data = sft_2.streamlines._data.astype(np.float16)
    dimensions = tuple(sft_1.dimensions)

    global matched_points_1, matched_points_2
    matched_points_1 = generate_matched_points(sft_1)
    matched_points_2 = generate_matched_points(sft_2)

    logging.info('Computing KDTree...')
    global tree_1, tree_2
    tree_1 = cKDTree(sft_1.streamlines._data)
    tree_2 = cKDTree(sft_2.streamlines._data)

    # Limits computation to mask AND streamlines (using density)
    if mask is None:
        mask = np.ones(dimensions)

    logging.info('Computing density maps...')
    sft_1.to_corner()
    sft_2.to_corner()
    density_1 = compute_tract_counts_map(sft_1.streamlines,
                                         dimensions).astype(float)
    density_2 = compute_tract_counts_map(sft_2.streamlines,
                                         dimensions).astype(float)
    mask = density_1 * density_2 * mask
    mask[mask > 0] = 1

    # Stop now if no overlap
    if np.count_nonzero(mask) == 0:
        logging.info("Bundles not overlapping! Not computing metrics.")
        acc_data = np.zeros(mask.shape) * np.nan
        corr_data = acc_data.copy()
        diff_data_norm = acc_data.copy()
        heatmap = acc_data.copy()
        return acc_data, corr_data, diff_data_norm, heatmap, mask

    logging.info('Computing correlation map... May be slow')
    corr_data = neighborhood_correlation_([density_1, density_2])
    corr_data[mask == 0] = np.nan

    logging.info('Computing TODI from tractogram #1...')
    global sh_data_1, sh_data_2
    todi_obj = TrackOrientationDensityImaging(dimensions, 'repulsion724')
    todi_obj.compute_todi(deepcopy(sft_1.streamlines), length_weights=True)
    todi_obj.mask_todi(mask)
    sh_data_1 = todi_obj.get_sh('descoteaux07', 8)
    sh_data_1 = todi_obj.reshape_to_3d(sh_data_1)
    sft_1.to_center()

    logging.info('Computing TODI from tractogram #2...')
    todi_obj = TrackOrientationDensityImaging(dimensions, 'repulsion724')
    todi_obj.compute_todi(deepcopy(sft_2.streamlines), length_weights=True)
    todi_obj.mask_todi(mask)
    sh_data_2 = todi_obj.get_sh('descoteaux07', 8)
    sh_data_2 = todi_obj.reshape_to_3d(sh_data_2)
    sft_2.to_center()

    global B
    B, _ = sh_to_sf_matrix(get_sphere('repulsion724'), 8, 'descoteaux07')

    diff_data, acc_data = _compare_tractogram_wrapper(
        mask, nbr_cpu, skip_streamlines_distance)

    # Normalize metrics and merge into a single heatmap
    diff_data_norm = normalize_metric(diff_data, reverse=True)
    heatmap = merge_metrics(acc_data, corr_data, diff_data_norm)

    return acc_data, corr_data, diff_data_norm, heatmap, mask
