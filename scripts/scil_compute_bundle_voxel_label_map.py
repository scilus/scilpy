#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute label image (Nifti) from bundle and centroid.
Each voxel will have the label of its nearest centroid point.

The number of labels will be the same as the centroid's number of points.
"""

import argparse
import itertools
import logging
import os

from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, set_sft_logger_level
from dipy.io.utils import is_header_compatible
from dipy.segment.clustering import qbx_and_merge
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial.ckdtree import cKDTree

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tracking.tools import resample_streamlines_num_points
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.tools import cut_outside_of_mask_streamlines
from scilpy.tractanalysis.distance_to_centroid import min_dist_to_centroid
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scilpy.utils.streamlines import uniformize_bundle_sft


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundles', nargs='+',
                   help='Fiber bundle file.')
    p.add_argument('in_centroid',
                   help='Centroid streamline corresponding to bundle.')
    p.add_argument('out_dir',
                   help='Directory to save all mapping and coloring file.')

    p.add_argument('--nb_pts', type=int,
                   help='Number of divisions for the bundles.\n'
                        'Default is the number of points of the centroid.')
    p.add_argument('--new_labeling', action='store_true',
                   help='.')
    p.add_argument('--min_streamline_count', type=int, default=100,
                   help='Minimum number of streamlines for filtering/cutting'
                        'operation [%(default)s].')
    p.add_argument('--min_voxel_count', type=int, default=1000,
                   help='Minimum number of voxels for filtering/cutting'
                        'operation [%(default)s].')
    p.add_argument('--colormap', default='jet',
                   help='Select the colormap for colored trk (data_per_point) '
                        '[%(default)s].')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def cube_correlation(density_list, binary_list, size=3):
    elem = np.arange(-(size//2), size//2 + 1).tolist()
    cube_ind = np.array(list(itertools.product(elem, elem, elem)))

    union = np.sum(binary_list, axis=0)
    corr_map = np.zeros(density_list[0].shape)
    indices = np.array(np.where(union)).T
    if len(density_list) > 1:
        for i, ind in enumerate(indices):
            ind = tuple(ind)

            cube_list = []
            for density in density_list:
                cube = map_coordinates(density, (cube_ind+ind).T, order=0)

                if np.count_nonzero(cube) > 1:
                    cube_list.append(cube.ravel())
            cov_matrix = np.triu(np.corrcoef(cube_list, cube_list), k=1)
            corr_map[ind] = np.average(cov_matrix[cov_matrix > 0])
    else:
        corr_map = binary_list[0]

    return corr_map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    set_sft_logger_level('ERROR')
    assert_inputs_exist(parser, args.in_bundles + [args.in_centroid],
                        optional=args.reference)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)

    sft_centroid = load_tractogram_with_reference(parser, args,
                                                  args.in_centroid)
    sft_centroid.to_vox()
    sft_centroid.to_corner()

    if len(sft_centroid.streamlines) < 1 \
            or len(sft_centroid.streamlines) > 1:
        logging.error('Centroid file {} should contain one streamline. '
                      'Skipping'.format(args.in_centroid))
        raise ValueError

    sft_list = []
    for filename in args.in_bundles:
        sft = load_tractogram_with_reference(parser, args, filename)
        if not len(sft.streamlines):
            logging.error('Empty bundle file {}. '
                          'Skipping'.format(args.in_bundle))
            raise ValueError
        sft.to_vox()
        sft.to_corner()
        sft_list.append(sft)

        if len(sft_list):
            if not is_header_compatible(sft_list[0], sft_list[-1]):
                parser.error('ERROR HEADER')

    density_list = []
    binary_list = []
    for sft in sft_list:
        density = compute_tract_counts_map(sft.streamlines,
                                           sft.dimensions).astype(float)
        binary = np.zeros(sft.dimensions)
        binary[density > 0] = 1
        binary_list.append(binary)

        density = gaussian_filter(density, 1) * binary
        density[binary < 1] += np.random.normal(0.0, 1.0,
                                                binary[binary < 1].shape)
        density_list.append(density)

    if not is_header_compatible(sft_centroid, sft_list[0]):
        raise IOError('{} and {}do not have a compatible header'.format(
            args.in_centroid, args.in_bundle))

    corr_map = cube_correlation(density_list, binary_list)
    # Slightly cut the bundle at the edgge to clean up single streamline voxels
    # with no neighbor. Remove isolated voxels to keep a single 'blob'
    binary_bundle = np.zeros(corr_map.shape, dtype=bool)
    binary_bundle[corr_map > 0.5] = 1
    min_streamlines_count = 0
    for sft in sft_list:
        min_streamlines_count = min(len(sft), min_streamlines_count)

    structure_cross = ndi.generate_binary_structure(3, 1)
    if np.count_nonzero(binary_bundle) > args.min_voxel_count \
            and min_streamlines_count > args.min_streamline_count:
        binary_bundle = ndi.binary_dilation(binary_bundle,
                                            structure=structure_cross)
        binary_bundle = ndi.binary_erosion(binary_bundle,
                                           structure=structure_cross,
                                           iterations=2)

    bundle_disjoint, _ = ndi.label(binary_bundle)
    unique, count = np.unique(bundle_disjoint, return_counts=True)
    val = unique[np.argmax(count[1:])+1]
    binary_bundle[bundle_disjoint != val] = 0

    corr_map = corr_map*binary_bundle
    nib.save(nib.Nifti1Image(corr_map, sft_list[0].affine),
             os.path.join(args.out_dir, 'corr_map.nii.gz'))

    # Chop off some streamlines
    concat_sft = StatefulTractogram.from_sft([], sft_list[0])
    for i in range(len(sft_list)):
        sft_list[i] = cut_outside_of_mask_streamlines(sft_list[i],
                                                      binary_bundle)
        if len(sft_list[i]):
            concat_sft += sft_list[i]

    if args.nb_pts is not None:
        sft_centroid = resample_streamlines_num_points(sft_centroid,
                                                       args.nb_pts)
    else:
        args.nb_pts = len(sft_centroid.streamlines[0])

    thresholds = [24, 18, 12, 6] if args.new_labeling else [200]
    clusters_map = qbx_and_merge(concat_sft.streamlines, thresholds,
                                 nb_pts=args.nb_pts, verbose=False,
                                 rng=np.random.RandomState(1))
    final_streamlines = []
    final_label = []
    final_dist = []
    for c, cluster in enumerate(clusters_map):
        tmp_sft = StatefulTractogram.from_sft([cluster.centroid], concat_sft)
        uniformize_bundle_sft(tmp_sft, ref_bundle=sft_centroid)
        cluster_centroid = tmp_sft.streamlines[0] if args.new_labeling \
            else sft_centroid.streamlines[0]
        cluster_streamlines = ArraySequence(cluster[:])
        # TODO N^2 growth in RAM, should split it if we want to do nb_pts = 100
        min_dist_label, min_dist = min_dist_to_centroid(cluster_streamlines._data,
                                                        cluster_centroid)
        min_dist_label += 1  # 0 means no labels

        # It is not allowed that labels jumps labels for consistency
        # Streamlines should have continous labels
        curr_ind = 0
        for i, streamline in enumerate(cluster_streamlines):
            next_ind = curr_ind + len(streamline)
            curr_labels = min_dist_label[curr_ind:next_ind]
            curr_dist = min_dist[curr_ind:next_ind]
            curr_ind = next_ind

            # Flip streamlines so the labels increase (facilitate if/else)
            # Should always be ordered in nextflow pipeline
            gradient = np.gradient(curr_labels)
            if len(np.argwhere(gradient < 0)) > len(np.argwhere(gradient > 0)):
                streamline = streamline[::-1]
                curr_labels = curr_labels[::-1]
                curr_dist = curr_dist[::-1]

            # Find jumps, cut them and find the longest
            gradient = np.ediff1d(curr_labels)
            max_jump = max(args.nb_pts // 5, 1)
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
                curr_dist = np.split(curr_dist,
                                     pos_jump)[max_pos]

            final_streamlines.append(streamline)
            final_label.append(curr_labels)
            final_dist.append(curr_dist)

    final_streamlines = ArraySequence(final_streamlines)
    labels_array = ArraySequence(final_label)
    dist_array = ArraySequence(final_dist)
    kd_tree = cKDTree(final_streamlines._data)
    img_labels = np.zeros(binary_bundle.shape, dtype=np.int16)
    img_distances = np.zeros(binary_bundle.shape, dtype=float)
    indices = np.nonzero(binary_bundle)

    for i in range(len(indices[0])):
        ind = np.array([indices[0][i], indices[1][i], indices[2][i]],
                       dtype=int)
        neighbor_ids = kd_tree.query_ball_point(ind, 2.0)
        if not neighbor_ids:
            continue
        labels_val = labels_array._data[neighbor_ids]
        dist_centro = dist_array._data[neighbor_ids]
        dist_vox = np.linalg.norm(final_streamlines._data[neighbor_ids] - ind,
                                  axis=1)

        if np.sum(dist_centro) > 0:
            img_labels[tuple(ind)] = np.round(
                np.average(labels_val, weights=dist_centro*dist_vox))
            img_distances[tuple(ind)] = np.average(dist_centro*dist_vox)
        else:
            img_labels[tuple(ind)] = np.round(
                np.average(labels_val, weights=dist_vox))
            img_distances[tuple(ind)] = np.average(dist_vox)

    for i, sft in enumerate(sft_list):
        sub_out_dir = os.path.join(args.out_dir, '_{}'.format(i))
        os.mkdir(sub_out_dir)
        if len(sft):
            tmp_labels = ndi.map_coordinates(img_labels,
                                             sft.streamlines._data.T-0.5,
                                             order=0)
            tmp_dists = ndi.map_coordinates(img_distances,
                                            sft.streamlines._data.T-0.5,
                                            order=0)
            tmp_corr = ndi.map_coordinates(corr_map,
                                           sft.streamlines._data.T-0.5,
                                           order=0)
        else:
            tmp_labels = []
            tmp_dists = []
            tmp_corr = []

        # Re-arrange the new cut streamlines and their metadata
        # Compute the voxels equivalent of the labels maps
        new_sft = StatefulTractogram.from_sft(sft.streamlines, sft_list[0])
        nib.save(nib.Nifti1Image(img_labels, sft_list[0].affine),
                 os.path.join(args.out_dir, 'labels_map.nii.gz'))
        nib.save(nib.Nifti1Image(img_distances, sft_list[0].affine),
                 os.path.join(args.out_dir, 'distances_map.nii.gz'))

        # WARNING: WILL NOT WORK WITH THE INPUT TRK !
        # These will fit only with the TRK saved below.
        np.savez_compressed(os.path.join(sub_out_dir, 'mapping_labels.npz'),
                            tmp_labels)
        np.savez_compressed(os.path.join(sub_out_dir, 'mapping_dists.npz'),
                            tmp_dists)
        np.savez_compressed(os.path.join(sub_out_dir, 'mapping_corr.npz'),
                            tmp_corr)

        if len(sft):
            cmap = plt.get_cmap(args.colormap)
            new_sft.data_per_point['color'] = ArraySequence(
                new_sft.streamlines)

        # Nicer visualisation for MI-Brain
        if len(sft):
            new_sft.data_per_point['color']._data = cmap(
                tmp_labels / np.max(tmp_labels))[:, 0:3] * 255
        save_tractogram(new_sft,
                        os.path.join(sub_out_dir, 'labels.trk'))

        if len(sft):
            new_sft.data_per_point['color']._data = cmap(
                tmp_dists / np.max(tmp_dists))[:, 0:3] * 255
        save_tractogram(new_sft,
                        os.path.join(sub_out_dir, 'dists.trk'))

        if len(sft):
            new_sft.data_per_point['color']._data = cmap(tmp_corr)[
                :, 0:3] * 255
        save_tractogram(new_sft,
                        os.path.join(sub_out_dir, 'corr.trk'))


if __name__ == '__main__':
    main()
