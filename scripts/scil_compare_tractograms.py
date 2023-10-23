#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script is designed to compare and help visualize differences between
two tractograms. Which can be especially useful in studies where multiple
tractograms from different algorithms or parameters need to be compared.

The difference is computed in terms of
- A voxel-wise spatial distance between streamlines, out_diff.nii.gz
- A correlation (ACC) between streamline orientation (TODI), out_acc.nii.gz
- A correlation between streamline density, out_corr.nii.gz
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import logging
import os
from tqdm import tqdm
import warnings

from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.segment.fss import FastStreamlineSearch
import nibabel as nib
import numpy as np
import numpy.ma as ma
from scipy.spatial import cKDTree

from scilpy.image.volume_math import correlation
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             add_processes_arg,
                             add_verbose_arg,
                             is_header_compatible_multiple_files,
                             load_tractogram_with_reference,
                             validate_nbr_processes)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.todi import TrackOrientationDensityImaging


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram_1',
                   help='Input tractogram 1.')
    p.add_argument('in_tractogram_2',
                   help='Input tractogram 2.')

    p.add_argument('--out_dir', default='',
                   help='Directory where all output files will be saved. '
                        '\nIf not specified, outputs will be saved in the current '
                        'directory.')
    p.add_argument('--out_prefix', default='out',
                   help='Prefix for output files. Useful for distinguishing between '
                        'different runs.')

    p.add_argument('--in_mask', metavar='IN_FILE',
                   help='Optional input mask.')

    add_processes_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def generate_matched_points(sft):
    """
    Generate an array where each element i is set to the index of the streamline
    that contributes the ith point.

    Parameters:
    -----------
    sft : StatefulTractogram
        The stateful tractogram containing the streamlines.

    Returns:
    --------
    matched_points : ndarray
        An array where each element is set to the index of the streamline
        that contributes that point.
    """
    total_points = sft.streamlines._data.shape[0]
    offsets = sft.streamlines._offsets

    matched_points = np.zeros(total_points, dtype=np.uint64)

    for i in range(len(offsets) - 1):
        matched_points[offsets[i]:offsets[i+1]] = i

    matched_points[offsets[-1]:] = len(offsets) - 1

    return matched_points


def compute_difference_for_voxel(chunk_indices):
    """
    Compute the difference for a single voxel index.
    """
    global sft_1, sft_2, matched_points_1, matched_points_2, tree_1, tree_2, \
        sh_data_1, sh_data_2
    results = []
    for vox_ind in chunk_indices:
        vox_ind = tuple(vox_ind)

        # Get the streamlines in the neighborhood (i.e., 2mm away)
        pts_ind_1 = tree_1.query_ball_point(vox_ind, 1.5)
        if not pts_ind_1:
            results.append([-1, -1])
            continue
        strs_ind_1 = np.unique(matched_points_1[pts_ind_1])
        neighb_streamlines_1 = sft_1.streamlines[strs_ind_1]

        # Get the streamlines in the neighborhood (i.e., 1mm away)
        pts_ind_2 = tree_2.query_ball_point(vox_ind, 1.5)
        if not pts_ind_2:
            results.append([-1, -1])
            continue
        strs_ind_2 = np.unique(matched_points_2[pts_ind_2])
        neighb_streamlines_2 = sft_2.streamlines[strs_ind_2]

        with warnings.catch_warnings(record=True) as _:
            fss = FastStreamlineSearch(neighb_streamlines_1, 10, resampling=12)
            dist_mat = fss.radius_search(neighb_streamlines_2, 10)
            sparse_dist_mat = np.abs(dist_mat.tocsr()).toarray()
            sparse_ma_dist_mat = np.ma.masked_where(sparse_dist_mat < 1e-3,
                                                    sparse_dist_mat)
            sparse_ma_dist_vec = np.squeeze(np.min(sparse_ma_dist_mat,
                                                   axis=0))

            if np.any(sparse_ma_dist_vec):
                global B
                sf_1 = np.dot(sh_data_1[vox_ind], B)
                sf_2 = np.dot(sh_data_2[vox_ind], B)
                dist = np.average(sparse_ma_dist_vec)
                corr = np.corrcoef(sf_1, sf_2)[0, 1]
                results.append([dist, corr])
            else:
                results.append([-1, -1])

    return results


def normalize_metric(metric, reverse=False):
    """
    Normalize a metric to be in the range [0, 1], ignoring specified values.
    """
    mask = np.isnan(metric)
    masked_metric = ma.masked_array(metric, mask)

    min_val, max_val = masked_metric.min(), masked_metric.max()
    normalized_metric = (masked_metric - min_val) / (max_val - min_val)

    if reverse:
        normalized_metric = 1 - normalized_metric

    return ma.filled(normalized_metric, fill_value=np.nan)


def merge_metrics(acc, corr, diff, beta=1.0):
    """
    Merge the three metrics into a single heatmap using a weighted geometric mean,
    ignoring specified values.
    """
    mask = np.isnan(acc) | np.isnan(corr) | np.isnan(diff)
    masked_acc, masked_corr, masked_diff = [
        ma.masked_array(x, mask) for x in [acc, corr, diff]]

    # Calculate the geometric mean for valid data
    geometric_mean = np.cbrt(masked_acc * masked_corr * masked_diff)

    # Apply a boosting factor
    boosted_mean = geometric_mean ** beta

    return ma.filled(boosted_mean, fill_value=np.nan)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.in_tractogram_1,
                                 args.in_tractogram_2])
    is_header_compatible_multiple_files(parser, [args.in_tractogram_1,
                                                 args.in_tractogram_2],
                                        verbose_all_compatible=True,
                                        reference=args.reference)

    if args.out_prefix and args.out_prefix[-1] == '_':
        args.out_prefix = args.out_prefix[:-1]
    out_corr_filename = os.path.join(args.out_dir,
                                     '{}_correlation.nii.gz'.format(args.out_prefix))
    out_acc_filename = os.path.join(args.out_dir,
                                    '{}_acc.nii.gz'.format(args.out_prefix))
    out_diff_filename = os.path.join(args.out_dir,
                                     '{}_diff.nii.gz'.format(args.out_prefix))
    out_merge_filename = os.path.join(args.out_dir,
                                      '{}_heatmap.nii.gz'.format(args.out_prefix))
    assert_output_dirs_exist_and_empty(parser, args, [], optional=args.out_dir)
    assert_outputs_exist(parser, args, [out_corr_filename,
                                        out_acc_filename,
                                        out_diff_filename,
                                        out_merge_filename])
    nbr_cpu = validate_nbr_processes(parser, args)

    logging.info('Loading tractograms...')
    global sft_1, sft_2
    sft_1 = load_tractogram_with_reference(parser, args, args.in_tractogram_1)
    # sft_1 = resample_streamlines_step_size(sft_1, 0.5)
    sft_2 = load_tractogram_with_reference(parser, args, args.in_tractogram_2)
    # sft_2 = resample_streamlines_step_size(sft_2, 0.5)
    sft_1.to_vox()
    sft_2.to_vox()
    sft_1.streamlines._data = sft_1.streamlines._data.astype(np.float16)
    sft_2.streamlines._data = sft_2.streamlines._data.astype(np.float16)
    affine, dimensions = sft_1.affine, sft_1.dimensions

    global matched_points_1, matched_points_2
    matched_points_1 = generate_matched_points(sft_1)
    matched_points_2 = generate_matched_points(sft_2)

    logging.info('Computing KDTree...')
    global tree_1, tree_2
    tree_1 = cKDTree(sft_1.streamlines._data)
    tree_2 = cKDTree(sft_2.streamlines._data)

    # Limits computation to mask AND streamlines (using density)
    if args.in_mask:
        mask = nib.load(args.in_mask).get_fdata()
    else:
        mask = np.ones(dimensions)

    logging.info('Computing density maps...')
    density_1 = compute_tract_counts_map(sft_1.streamlines,
                                         dimensions).astype(float)
    density_2 = compute_tract_counts_map(sft_2.streamlines,
                                         dimensions).astype(float)
    mask = density_1 * density_2 * mask
    mask[mask > 0] = 1

    logging.info('Computing correlation map...')
    corr_data = correlation([density_1, density_2], None) * mask
    nib.save(nib.Nifti1Image(corr_data, affine), out_corr_filename)

    logging.info('Computing TODI #1...')
    global sh_data_1, sh_data_2
    sft_1.to_corner()
    todi_obj = TrackOrientationDensityImaging(
        tuple(dimensions), 'repulsion724')
    todi_obj.compute_todi(deepcopy(sft_1.streamlines), length_weights=True)
    todi_obj.mask_todi(mask)
    sh_data_1 = todi_obj.get_sh('descoteaux07', 8)
    sh_data_1 = todi_obj.reshape_to_3d(sh_data_1)
    sft_1.to_center()

    logging.info('Computing TODI #2...')
    sft_2.to_corner()
    todi_obj = TrackOrientationDensityImaging(
        tuple(dimensions), 'repulsion724')
    todi_obj.compute_todi(deepcopy(sft_2.streamlines), length_weights=True)
    todi_obj.mask_todi(mask)
    sh_data_2 = todi_obj.get_sh('descoteaux07', 8)
    sh_data_2 = todi_obj.reshape_to_3d(sh_data_2)
    sft_2.to_center()

    global B
    B, _ = sh_to_sf_matrix(get_sphere('repulsion724'), 8, 'descoteaux07')

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
            compute_difference_for_voxel, chunk): chunk for chunk in index_chunks}

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

    logging.info('Saving results...')
    nib.save(nib.Nifti1Image(diff_data, affine), out_diff_filename)
    nib.save(nib.Nifti1Image(acc_data, affine), out_acc_filename)

    # Normalize metrics
    acc_norm = normalize_metric(acc_data)
    corr_norm = normalize_metric(corr_data)
    diff_norm = normalize_metric(diff_data, reverse=True)
    indices_minus_one = np.where((acc_data == -1) | (corr_data == -1) |
                                 (diff_data == -1))

    # Merge into a single heatmap
    heatmap = merge_metrics(acc_norm, corr_norm, diff_norm)

    # Save as a new NIFTI file
    heatmap[indices_minus_one] = np.nan
    nib.save(nib.Nifti1Image(heatmap, affine), out_merge_filename)


if __name__ == "__main__":
    main()
