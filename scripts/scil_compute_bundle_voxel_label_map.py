#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute label image (Nifti) from bundle and centroid.
Each voxel will have the label of its nearest centroid point.

The number of labels will be the same as the centroid's number of points.
"""

import argparse
import logging

from dipy.align.streamlinear import (BundleMinDistanceMetric,
                                     StreamlineLinearRegistration)
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, set_sft_logger_level
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
    p.add_argument('out_distances_map',
                   help='Nifti image showing distances to centroids.')

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


def _affine_slr(sft_bundle, sft_centroid):
    x0 = np.zeros((7,))
    bounds_dof = [(-10, 10), (-10, 10), (-10, 10),
                  (-5, 5), (-5, 5), (-5, 5),
                  (0.95, 1.05)]
    metric = BundleMinDistanceMetric(num_threads=1)
    slr = StreamlineLinearRegistration(metric=metric, method="L-BFGS-B",
                                       bounds=bounds_dof, x0=x0,
                                       num_threads=1)
    tmp_bundle = set_number_of_points(sft_bundle.streamlines.copy(), 20)
    tmp_centroid = set_number_of_points(sft_centroid.streamlines.copy(), 20)
    slm = slr.optimize(tmp_bundle, tmp_centroid)
    sft_centroid.streamlines = transform_streamlines(sft_centroid.streamlines,
                                                     slm.matrix)
    return sft_centroid


def _distance_using_mask(sft_bundle, binary_centroid):
    binary_bundle = compute_tract_counts_map(sft_bundle.streamlines,
                                             sft_bundle.dimensions).astype(
        np.bool)

    # Iteratively dilate the mask until the cleaned bundle mask is filled
    count = 1
    min_distances = np.zeros(sft_bundle.dimensions)
    last_count = 0
    while np.count_nonzero(binary_centroid) != np.count_nonzero(binary_bundle):
        previous_centroid = binary_centroid.copy()
        binary_centroid = ndi.binary_dilation(binary_centroid,
                                              structure=np.ones((3, 3, 3)))

        # Must follow the curve of the bundle (gyri/sulci)
        binary_centroid *= binary_bundle
        tmp_binary_centroid = binary_centroid.copy()
        tmp_binary_centroid[previous_centroid] = 0
        min_distances[tmp_binary_centroid > 0] = count
        count += 1
        if last_count == np.count_nonzero(binary_centroid):
            break
        last_count = np.count_nonzero(binary_centroid)

    sft_bundle.to_center()
    distances_arr_seq = ArraySequence()
    distances_arr_seq._data = ndi.map_coordinates(min_distances,
                                                  sft_bundle.streamlines._data.T,
                                                  order=0)
    distances_arr_seq._data /= np.max(distances_arr_seq._data)
    distances_arr_seq._offsets = sft_bundle.streamlines._offsets
    distances_arr_seq._lengths = sft_bundle.streamlines._lengths

    return distances_arr_seq, min_distances


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    set_sft_logger_level('ERROR')
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
    if np.count_nonzero(binary_bundle) > 10000:
        binary_bundle = ndi.binary_dilation(binary_bundle,
                                            structure=np.ones((3, 3, 3)))
        binary_bundle = ndi.binary_erosion(binary_bundle,
                                           structure=structure, iterations=2)

    bundle_disjoint, _ = ndi.label(binary_bundle)
    unique, count = np.unique(bundle_disjoint, return_counts=True)
    val = unique[np.argmax(count[1:])+1]
    binary_bundle[bundle_disjoint != val] = 0

    # Chop off some streamlines
    cut_sft = cut_outside_of_mask_streamlines(sft_bundle, binary_bundle)

    if args.nb_pts is not None:
        sft_centroid = resample_streamlines_num_points(sft_centroid,
                                                       args.nb_pts)
    else:
        args.nb_pts = len(sft_centroid.streamlines[0])

    # Generate a centroids labels mask for the centroid alone
    sft_centroid.to_vox()
    sft_centroid.to_corner()
    sft_centroid = _affine_slr(sft_bundle, sft_centroid)

    # Map every streamlines points to the centroids
    binary_centroid = compute_tract_counts_map(sft_centroid.streamlines,
                                               sft_centroid.dimensions).astype(
                                                   np.bool)
    # TODO N^2 growth in RAM, should split it if we want to do nb_pts = 100
    min_dist_label, min_dist = min_dist_to_centroid(cut_sft.streamlines._data,
                                                    sft_centroid.streamlines._data)
    min_dist_label += 1  # 0 means no labels

    # It is not allowed that labels jumps labels for consistency
    # Streamlines should have continous labels
    curr_ind = 0
    final_streamlines = []
    final_label = []
    final_dist = []
    for i, streamline in enumerate(cut_sft.streamlines):
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

    # Re-arrange the new cut streamlines and their metadata
    # Compute the voxels equivalent of the labels maps
    new_sft = StatefulTractogram.from_sft(final_streamlines, sft_bundle)
    labels_array = ArraySequence(final_label)

    tdi_mask_nzr = np.nonzero(binary_bundle)
    tdi_mask_nzr_ind = np.transpose(tdi_mask_nzr)
    min_dist_ind, _ = min_dist_to_centroid(tdi_mask_nzr_ind,
                                           sft_centroid.streamlines[0])
    img_labels = np.zeros(binary_centroid.shape, dtype=np.int16)
    img_labels[tdi_mask_nzr] = min_dist_ind + 1  # 0 is background value

    # Approximation of the distance using the WM diffusion approach
    # In non-obstructed line, equivalent to euclidian distance
    distances_array, img_distances = _distance_using_mask(new_sft,
                                                          binary_centroid)

    nib.save(nib.Nifti1Image(img_labels, sft_bundle.affine),
             args.out_labels_map)
    nib.save(nib.Nifti1Image(img_distances, sft_bundle.affine),
             args.out_distances_map)

    if args.labels_color_dpp or args.distances_color_dpp \
            or args.out_labels_npz or args.out_distances_npz:
        # WARNING: WILL NOT WORK WITH THE INPUT TRK !
        # These will fit only with the TRK saved below.
        if args.out_labels_npz:
            np.savez_compressed(args.out_labels_npz, labels_array._data)
        if args.out_distances_npz:
            np.savez_compressed(args.out_distances_npz, labels_array._data)

        cmap = plt.get_cmap(args.colormap)
        new_sft.data_per_point['color'] = ArraySequence(new_sft.streamlines)

        # Nicer visualisation for MI-Brain
        if args.labels_color_dpp:
            new_sft.data_per_point['color']._data = cmap(
                labels_array._data / np.max(labels_array._data))[:, 0:3] * 255
            save_tractogram(new_sft, args.labels_color_dpp)

        if args.distances_color_dpp:
            new_sft.data_per_point['color']._data = cmap(
                distances_array._data / np.max(distances_array._data))[:, 0:3] * 255
            save_tractogram(new_sft, args.distances_color_dpp)


if __name__ == '__main__':
    main()
