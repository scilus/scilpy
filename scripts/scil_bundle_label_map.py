#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the label image (Nifti) from a centroid and tractograms (all
representing the same bundle). The label image represents the coverage of
the bundle, segmented into regions labelled from 0 to --nb_pts, starting from
the head, ending in the tail.

Each voxel will have the label of its nearest centroid point.

The number of labels will be the same as the centroid's number of points.

Formerly: scil_compute_bundle_voxel_label_map.py
"""

import argparse
import logging
import os

from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.utils import is_header_compatible
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree

from scilpy.image.volume_math import neighborhood_correlation_
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.bundle_operations import uniformize_bundle_sft
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.distance_to_centroid import min_dist_to_centroid
from scilpy.tractograms.streamline_and_mask_operations import \
    cut_outside_of_mask_streamlines
from scilpy.tractograms.streamline_operations import \
    resample_streamlines_num_points
from scilpy.viz.color import get_lookup_table


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundles', nargs='+',
                   help='Fiber bundle file.')
    p.add_argument('in_centroid',
                   help='Centroid streamline corresponding to bundle.')
    p.add_argument('out_dir',
                   help='Directory to save all mapping and coloring files:\n'
                        '  - correlation_map.nii.gz\n'
                        '  - session_x/labels_map.nii.gz\n'
                        '  - session_x/distance_map.nii.gz\n'
                        '  - session_x/correlation_map.nii.gz\n'
                        '  - session_x/labels.trk\n'
                        '  - session_x/distance.trk\n'
                        '  - session_x/correlation.trk\n'
                        'Where session_x is numbered with each bundle.')

    p.add_argument('--nb_pts', type=int,
                   help='Number of divisions for the bundles.\n'
                        'Default is the number of points of the centroid.')
    p.add_argument('--colormap', default='jet',
                   help='Select the colormap for colored trk (data_per_point) '
                        '[%(default)s].')
    p.add_argument('--new_labelling', action='store_true',
                   help='Use the new labelling method (multi-centroids).')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bundles + [args.in_centroid],
                        optional=args.reference)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)

    sft_centroid = load_tractogram_with_reference(parser, args,
                                                  args.in_centroid)

    sft_centroid.to_vox()
    sft_centroid.to_corner()

    sft_list = []
    for filename in args.in_bundles:
        sft = load_tractogram_with_reference(parser, args, filename)
        if not len(sft.streamlines):
            raise IOError('Empty bundle file {}. '
                          'Skipping'.format(args.in_bundle))
        sft.to_vox()
        sft.to_corner()
        sft_list.append(sft)

        if len(sft_list):
            if not is_header_compatible(sft_list[0], sft_list[-1]):
                parser.error('Header of {} and {} are not compatible'.format(
                    args.in_bundles[0], filename))

    density_list = []
    binary_list = []
    for sft in sft_list:
        density = compute_tract_counts_map(sft.streamlines,
                                           sft.dimensions).astype(float)
        binary = np.zeros(sft.dimensions)
        binary[density > 0] = 1
        binary_list.append(binary)
        density_list.append(density)

    if not is_header_compatible(sft_centroid, sft_list[0]):
        raise IOError('{} and {}do not have a compatible header'.format(
            args.in_centroid, args.in_bundle))

    if len(density_list) > 1:
        corr_map = neighborhood_correlation_(density_list)
    else:
        corr_map = density_list[0].astype(float)
        corr_map[corr_map > 0] = 1

    # Slightly cut the bundle at the edge to clean up single streamline voxels
    # with no neighbor. Remove isolated voxels to keep a single 'blob'
    binary_bundle = np.zeros(corr_map.shape, dtype=bool)
    binary_bundle[corr_map > 0.5] = 1

    bundle_disjoint, _ = ndi.label(binary_bundle)
    unique, count = np.unique(bundle_disjoint, return_counts=True)
    val = unique[np.argmax(count[1:])+1]
    binary_bundle[bundle_disjoint != val] = 0

    corr_map = corr_map*binary_bundle
    nib.save(nib.Nifti1Image(corr_map, sft_list[0].affine),
             os.path.join(args.out_dir, 'correlation_map.nii.gz'))

    # Chop off some streamlines
    concat_sft = StatefulTractogram.from_sft([], sft_list[0])
    for i in range(len(sft_list)):
        sft_list[i] = cut_outside_of_mask_streamlines(sft_list[i],
                                                      binary_bundle)
        if len(sft_list[i]):
            concat_sft += sft_list[i]

    args.nb_pts = len(sft_centroid.streamlines[0]) if args.nb_pts is None \
        else args.nb_pts

    sft_centroid = resample_streamlines_num_points(sft_centroid, args.nb_pts)
    tmp_sft = resample_streamlines_num_points(concat_sft, args.nb_pts)

    if not args.new_labelling:
        new_streamlines = sft_centroid.streamlines.copy()
        sft_centroid = StatefulTractogram.from_sft([new_streamlines[0]],
                                                   sft_centroid)
    else:
        srr = StreamlineLinearRegistration()
        srm = srr.optimize(static=tmp_sft.streamlines,
                           moving=sft_centroid.streamlines)
        sft_centroid.streamlines = srm.transform(sft_centroid.streamlines)

    uniformize_bundle_sft(concat_sft, ref_bundle=sft_centroid[0])
    labels, dists = min_dist_to_centroid(concat_sft.streamlines._data,
                                         sft_centroid.streamlines._data,
                                         args.nb_pts)
    labels += 1  # 0 means no labels

    # It is not allowed that labels jumps labels for consistency
    # Streamlines should have continous labels
    final_streamlines = []
    final_label = []
    final_dists = []
    curr_ind = 0
    for i, streamline in enumerate(concat_sft.streamlines):
        next_ind = curr_ind + len(streamline)
        curr_labels = labels[curr_ind:next_ind]
        curr_dists = dists[curr_ind:next_ind]
        curr_ind = next_ind

        # Flip streamlines so the labels increase (facilitate if/else)
        # Should always be ordered in nextflow pipeline
        gradient = np.gradient(curr_labels)
        if len(np.argwhere(gradient < 0)) > len(np.argwhere(gradient > 0)):
            streamline = streamline[::-1]
            curr_labels = curr_labels[::-1]
            curr_dists = curr_dists[::-1]

        # # Find jumps, cut them and find the longest
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
            curr_dists = np.split(curr_dists,
                                  pos_jump)[max_pos]

        final_streamlines.append(streamline)
        final_label.append(curr_labels)
        final_dists.append(curr_dists)

    final_streamlines = ArraySequence(final_streamlines)
    final_labels = ArraySequence(final_label)
    final_dists = ArraySequence(final_dists)

    kd_tree = cKDTree(final_streamlines._data)
    labels_map = np.zeros(binary_bundle.shape, dtype=np.int16)
    distance_map = np.zeros(binary_bundle.shape, dtype=float)
    indices = np.array(np.nonzero(binary_bundle), dtype=int).T

    for ind in indices:
        _, neighbor_ids = kd_tree.query(ind, k=5)

        if not len(neighbor_ids):
            continue

        labels_val = final_labels._data[neighbor_ids]
        dists_val = final_dists._data[neighbor_ids]
        sum_dists_vox = np.sum(dists_val)
        weights_vox = np.exp(-dists_val / sum_dists_vox)

        vote = np.bincount(labels_val, weights=weights_vox)
        total = np.arange(np.amax(labels_val+1))
        winner = total[np.argmax(vote)]
        labels_map[ind[0], ind[1], ind[2]] = winner
        distance_map[ind[0], ind[1], ind[2]] = np.average(dists_val)

        cmap = get_lookup_table(args.colormap)

    for i, sft in enumerate(sft_list):
        if len(sft_list) > 1:
            sub_out_dir = os.path.join(args.out_dir, 'session_{}'.format(i+1))
        else:
            sub_out_dir = args.out_dir
        new_sft = StatefulTractogram.from_sft(sft.streamlines, sft_list[0])
        if not os.path.isdir(sub_out_dir):
            os.mkdir(sub_out_dir)

        # Save each session map if multiple inputs
        nib.save(nib.Nifti1Image((binary_list[i]*labels_map).astype(np.uint16),
                                 sft_list[0].affine),
                 os.path.join(sub_out_dir, 'labels_map.nii.gz'))
        nib.save(nib.Nifti1Image(binary_list[i]*distance_map,
                                 sft_list[0].affine),
                 os.path.join(sub_out_dir, 'distance_map.nii.gz'))
        nib.save(nib.Nifti1Image(binary_list[i]*corr_map,
                                 sft_list[0].affine),
                 os.path.join(sub_out_dir, 'correlation_map.nii.gz'))

        if len(sft):
            tmp_labels = ndi.map_coordinates(labels_map,
                                             sft.streamlines._data.T-0.5,
                                             order=0)
            tmp_dists = ndi.map_coordinates(distance_map,
                                            sft.streamlines._data.T-0.5,
                                            order=0)
            tmp_corr = ndi.map_coordinates(corr_map,
                                           sft.streamlines._data.T-0.5,
                                           order=0)
            cmap = plt.colormaps[args.colormap]
            new_sft.data_per_point['color'] = ArraySequence(
                new_sft.streamlines)

            # Nicer visualisation for MI-Brain
            new_sft.data_per_point['color']._data = cmap(
                tmp_labels / np.max(tmp_labels))[:, 0:3] * 255
        save_tractogram(new_sft,
                        os.path.join(sub_out_dir, 'labels.trk'))

        if len(sft):
            new_sft.data_per_point['color']._data = cmap(
                tmp_dists / np.max(tmp_dists))[:, 0:3] * 255
        save_tractogram(new_sft,
                        os.path.join(sub_out_dir, 'distance.trk'))

        if len(sft):
            new_sft.data_per_point['color']._data = cmap(tmp_corr)[
                :, 0:3] * 255
        save_tractogram(new_sft,
                        os.path.join(sub_out_dir, 'correlation.trk'))


if __name__ == '__main__':
    main()
