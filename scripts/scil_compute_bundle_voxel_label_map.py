#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute label image (Nifti) from bundle and centroid.
Each voxel will have the label of its nearest centroid point.

The number of labels will be the same as the centroid's number of points.
"""

import argparse
from itertools import product
import logging

import dijkstra3d
from dipy.align.streamlinear import (BundleMinDistanceMetric,
                                     StreamlineLinearRegistration)
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamline import transform_streamlines, set_number_of_points
from dipy.io.utils import is_header_compatible
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
import scipy.ndimage as ndi

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_verbose_arg)
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
    p.add_argument('out_distances_map',
                   help='Nifti image showing distances to centroids.')

    p.add_argument('--nb_pts', type=int,
                   help='Number of divisions for the bundles.')

    p.add_argument('--out_labels_npz', metavar='FILE',
                   help='File mapping of points to labels.')
    p.add_argument('--out_distances_npz', metavar='FILE',
                   help='File mapping of points to distances.')

    p.add_argument('--labels_color_dpp', metavar='FILE',
                   help='Save a trk with labels color indicating labels.')
    p.add_argument('--distances_color_dpp', metavar='FILE',
                   help='Save a trk with labels color indicating labels.')
    p.add_argument('--colormap', default='jet',
                   help='Select the colormap for colored trk (data_per_point) '
                        '[%(default)s].')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def _rigid_slr(sft_bundle, sft_centroid):
    bounds_dof = [(-10, 10), (-10, 10), (-10, 10),
                  (-5, 5), (-5, 5), (-5, 5)]
    metric = BundleMinDistanceMetric(num_threads=1)
    slr = StreamlineLinearRegistration(metric=metric, method="Powell",
                                       bounds=bounds_dof,
                                       num_threads=1)
    tmp_bundle = set_number_of_points(sft_bundle.streamlines.copy(), 20)
    tmp_centroid = set_number_of_points(sft_centroid.streamlines.copy(), 20)
    slm = slr.optimize(tmp_bundle, tmp_centroid)
    sft_centroid.streamlines = transform_streamlines(sft_centroid.streamlines,
                                                     slm.matrix)
    return sft_centroid


def _get_neighbors_vote(pos, data):
    neighbors = list(product((1, -1), repeat=3))
    neighbors = np.array(neighbors, dtype=np.int32) + pos
    neighbors_val = ndi.map_coordinates(data, neighbors.T, order=0)
    unique, count = np.unique(neighbors_val, return_counts=True)
    # print(unique)
    if len(unique) > 1:
        return unique[np.argmax(unique[1:])+1]
    else:
        return 0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser,
                        [args.in_bundle, args.in_centroid],
                        optional=args.reference)
    assert_outputs_exist(parser, args, [args.out_labels_map,
                                        args.out_distances_map],
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

    if not len(sft_centroid.streamlines):
        logging.error('Centroid file {} should contain one streamline. '
                      'Skipping'.format(args.in_centroid))
        raise ValueError

    if not is_header_compatible(sft_centroid, sft_bundle):
        raise IOError('{} and {}do not have a compatible header'.format(
            args.in_centroid, args.in_bundle))

    sft_bundle.to_vox()
    sft_bundle.to_corner()

    binary_bundle = compute_tract_counts_map(sft_bundle.streamlines,
                                             sft_bundle.dimensions).astype(np.bool)

    structure = ndi.generate_binary_structure(3, 1)
    binary_bundle = ndi.binary_dilation(binary_bundle,
                                        structure=np.ones((3, 3, 3)))
    binary_bundle = ndi.binary_erosion(binary_bundle,
                                       structure=structure, iterations=2)

    bundle_disjoint, _ = ndi.label(binary_bundle)
    unique, count = np.unique(bundle_disjoint, return_counts=True)
    val = unique[np.argmax(count[1:])+1]
    binary_bundle[bundle_disjoint != val] = 0

    if args.nb_pts is not None:
        sft_centroid = resample_streamlines_num_points(sft_centroid,
                                                       args.nb_pts)
    sft_centroid.to_vox()
    sft_centroid.to_corner()
    sft_centroid = _rigid_slr(sft_bundle, sft_centroid)

    binary_centroid = compute_tract_counts_map(sft_centroid.streamlines,
                                               sft_centroid.dimensions).astype(np.bool)

    tdi_mask_nzr = np.nonzero(binary_centroid)
    tdi_mask_nzr_ind = np.transpose(tdi_mask_nzr)
    min_dist_ind, _ = min_dist_to_centroid(tdi_mask_nzr_ind,
                                           sft_centroid.streamlines[0])

    labels_mask = np.zeros(binary_centroid.shape)
    labels_mask[tdi_mask_nzr] = min_dist_ind + 1  # 0 is background value
    real_min_distances = np.ones(sft_bundle.dimensions, dtype=np.int16) * -1
    real_min_distances[labels_mask > 0] += 1

    count = 1
    while np.count_nonzero(labels_mask) != 22276:
        print('==', np.count_nonzero(labels_mask), np.count_nonzero(binary_bundle))
        closest_labels = np.zeros(sft_bundle.dimensions, dtype=np.uint16)
        # min_distances = np.ones(sft_bundle.dimensions, dtype=np.float32) * 9999
        previous_centroid = binary_centroid.copy()
        binary_centroid = ndi.binary_dilation(binary_centroid,
                                                   structure=np.ones((3,3,3)))
        binary_centroid *= binary_bundle

        tmp_binary_centroid = binary_centroid.copy()
        if count > 1:
            tmp_binary_centroid[previous_centroid] = 0
        positions = np.argwhere(tmp_binary_centroid)
        for j, ind_t in enumerate(np.argwhere(tmp_binary_centroid)):
            ind_t = tuple(ind_t)
            closest_labels[ind_t] = _get_neighbors_vote(ind_t, labels_mask)

        labels_mask[closest_labels > 0] = closest_labels[closest_labels > 0]
        real_min_distances[closest_labels > 0] = count
        # nib.save(nib.Nifti1Image(tmp_binary_centroid.astype(np.uint8), sft_bundle.affine),
        #             'lol.nii.gz')
        count += 1
        # if count == 15:
        #     break

    # SAVING
    nib.save(nib.Nifti1Image(labels_mask, sft_bundle.affine),
                args.out_labels_map)
    nib.save(nib.Nifti1Image(real_min_distances, sft_bundle.affine),
                args.out_distances_map)
    if args.labels_color_dpp or args.distance_color_dpp \
            or args.out_labels_npz or args.out_distances_npz:

        cut_sft = cut_outside_of_mask_streamlines(sft_bundle, binary_bundle)
        cut_sft.to_center()
        labels_array = ndi.map_coordinates(labels_mask,
                                            cut_sft.streamlines._data.T,
                                            order=0)
        distances_array = ndi.map_coordinates(real_min_distances,
                                                cut_sft.streamlines._data.T,
                                                order=0)
        if args.out_labels_npz:
            np.savez_compressed(labels_array, args.out_labels_npz)
        if args.out_distances_npz:
            np.savez_compressed(labels_array, args.out_distances_npz)

        cmap = plt.get_cmap(args.colormap)
        cut_sft.data_per_point['color'] = ArraySequence(
            cut_sft.streamlines)

        if args.labels_color_dpp:
            cut_sft.data_per_point['color']._data = cmap(
                labels_array / args.nb_pts)[:, 0:3] * 255
            # save_tractogram(cut_sft, 'labels_{}.trk'.format(count))
            save_tractogram(cut_sft, args.labels_color_dpp)

        if args.distances_color_dpp:
            cut_sft.data_per_point['color']._data = cmap(
                distances_array / np.max(distances_array))[:, 0:3] * 255
            # save_tractogram(cut_sft, 'distance{}.trk'.format(count))
            save_tractogram(cut_sft, args.distances_color_dpp)


if __name__ == '__main__':
    main()
