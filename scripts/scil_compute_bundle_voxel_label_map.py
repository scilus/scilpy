#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute label image (Nifti) from bundle and centroid.
Each voxel will have the label of its nearest centroid point.

The number of labels will be the same as the centroid's number of points.
"""

import argparse
import logging
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
                             assert_outputs_exist)
from scilpy.tracking.tools import resample_streamlines_num_points
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.tools import cut_outside_of_mask_streamlines
from scilpy.tractanalysis.distance_to_centroid import min_dist_to_centroid


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')
    p.add_argument('in_centroid',
                   help='Centroid streamline corresponding to bundle.')
    p.add_argument('out_labels_map',
                   help='Nifti image with corresponding labels.')

    p.add_argument('--nb_pts', type=int,
                   help='Number of divisions for the bundles.\n'
                        'Default is the number of points of the centroid.')

    p.add_argument('--out_labels_npz', metavar='FILE',
                   help='File mapping of points to labels.')
    p.add_argument('--out_distances_npz', metavar='FILE',
                   help='File mapping of points to distances.')

    p.add_argument('--labels_color_dpp', metavar='FILE',
                   help='Save bundle with labels coloring (.trk).')
    p.add_argument('--distances_color_dpp', metavar='FILE',
                   help='Save bundle with distances coloring (.trk).')
    p.add_argument('--colormap', default='jet',
                   help='Select the colormap for colored trk (data_per_point) '
                        '[%(default)s].')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    set_sft_logger_level('ERROR')
    assert_inputs_exist(parser,
                        [args.in_bundle, args.in_centroid],
                        optional=args.reference)
    assert_outputs_exist(parser, args, args.out_labels_map,
                         optional=[args.out_labels_npz,
                                   args.out_distances_npz,
                                   args.labels_color_dpp,
                                   args.distances_color_dpp])

    sft_bundle = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft_centroid = load_tractogram_with_reference(parser, args,
                                                  args.in_centroid)

    if not len(sft_bundle.streamlines):
        logging.error('Empty bundle file {}. '
                      'Skipping'.format(args.in_bundle))
        raise ValueError

    if len(sft_centroid.streamlines) < 1 \
            or len(sft_centroid.streamlines) > 1:
        logging.error('Centroid file {} should contain one streamline. '
                      'Skipping'.format(args.in_centroid))
        raise ValueError

    if not is_header_compatible(sft_centroid, sft_bundle):
        raise IOError('{} and {}do not have a compatible header'.format(
            args.in_centroid, args.in_bundle))

    sft_bundle.to_vox()
    sft_bundle.to_corner()

    # Slightly cut the bundle at the edgge to clean up single streamline voxels
    # with no neighbor. Remove isolated voxels to keep a single 'blob'
    binary_bundle = compute_tract_counts_map(sft_bundle.streamlines,
                                             sft_bundle.dimensions).astype(
                                                 np.bool)

    structure = ndi.generate_binary_structure(3, 1)
    if len(sft_bundle) > 1000:
        if np.count_nonzero(binary_bundle) > 10000:
            binary_bundle = ndi.binary_dilation(binary_bundle,
                                                structure=np.ones((3, 3, 3)))
            binary_bundle = ndi.binary_erosion(binary_bundle,
                                               structure=structure,
                                               iterations=2)

        bundle_disjoint, _ = ndi.label(binary_bundle)
        unique, count = np.unique(bundle_disjoint, return_counts=True)
        val = unique[np.argmax(count[1:])+1]
        binary_bundle[bundle_disjoint != val] = 0

        # Chop off some streamlines
        cut_sft = cut_outside_of_mask_streamlines(sft_bundle, binary_bundle)
    else:
        cut_sft = sft_bundle

    if args.nb_pts is not None:
        sft_centroid = resample_streamlines_num_points(sft_centroid,
                                                       args.nb_pts)
    else:
        args.nb_pts = len(sft_centroid.streamlines[0])

    clusters_map = qbx_and_merge(cut_sft.streamlines, [30, 20, 10, 5],
                                 nb_pts=args.nb_pts, verbose=False)
    final_streamlines = []
    final_label = []
    final_dist = []
    for c, cluster in enumerate(clusters_map):
        cluster_centroid = cluster.centroid
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
    img_distances = np.zeros(binary_bundle.shape, dtype=np.float32)
    indices = np.nonzero(binary_bundle)

    for i in range(len(indices[0])):
        ind = np.array([indices[0][i], indices[1][i], indices[2][i]],
                       dtype=np.int32)
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
    final_labels = ndi.map_coordinates(img_labels,
                                       final_streamlines._data.T-0.5,
                                       order=0)
    final_dists = ndi.map_coordinates(img_distances,
                                      final_streamlines._data.T-0.5,
                                      order=0)

    # Re-arrange the new cut streamlines and their metadata
    # Compute the voxels equivalent of the labels maps
    new_sft = StatefulTractogram.from_sft(final_streamlines, sft_bundle)

    nib.save(nib.Nifti1Image(img_labels, sft_bundle.affine),
             args.out_labels_map)

    if args.labels_color_dpp or args.distances_color_dpp \
            or args.out_labels_npz or args.out_distances_npz:
        # WARNING: WILL NOT WORK WITH THE INPUT TRK !
        # These will fit only with the TRK saved below.
        if args.out_labels_npz:
            np.savez_compressed(args.out_labels_npz, final_labels)
        if args.out_distances_npz:
            np.savez_compressed(args.out_distances_npz, final_dists)

        cmap = plt.get_cmap(args.colormap)
        new_sft.data_per_point['color'] = ArraySequence(new_sft.streamlines)

        # Nicer visualisation for MI-Brain
        if args.labels_color_dpp:
            new_sft.data_per_point['color']._data = cmap(
                final_labels / np.max(final_labels))[:, 0:3] * 255
            save_tractogram(new_sft, args.labels_color_dpp)

        if args.distances_color_dpp:
            new_sft.data_per_point['color']._data = cmap(
                final_dists / np.max(final_dists))[:, 0:3] * 255
            save_tractogram(new_sft, args.distances_color_dpp)


if __name__ == '__main__':
    main()
