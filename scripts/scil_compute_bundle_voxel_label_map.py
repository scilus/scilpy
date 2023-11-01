#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute label image (Nifti) from bundle and centroid.
Each voxel will have the label of its nearest centroid point.

The number of labels will be the same as the centroid's number of points.
"""

from sklearn.svm import SVC
from collections import defaultdict
from time import time
import argparse
import os

# from dipy.align.streamlinear import BundleMinDistanceMetric, StreamlineLinearRegistration
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, set_sft_logger_level, Space, Origin
from dipy.io.utils import is_header_compatible
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion

from scilpy.image.volume_math import correlation
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.distance_to_centroid import (min_dist_to_centroid,
                                                       compute_euclidean_barycenters,
                                                       compute_shell_barycenters,
                                                       transfer_and_diffuse_labels)
from scilpy.tractograms.streamline_and_mask_operations import \
    cut_outside_of_mask_streamlines
from scilpy.tractograms.streamline_operations import resample_streamlines_num_points
from scilpy.tractograms.streamline_and_mask_operations import \
    get_head_tail_density_maps
from scilpy.utils.streamlines import uniformize_bundle_sft
from scilpy.viz.utils import get_colormap


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
    p.add_argument('--colormap', default='jet',
                   help='Select the colormap for colored trk (data_per_point) '
                        '[%(default)s].')
    p.add_argument('--new_labelling', action='store_true',
                   help='Use the new labelling method (multi-centroids).')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    t0 = time()
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
    print('Loading time', time()-t0)
    t0 = time()
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
        corr_map = correlation(density_list, None)
    else:
        corr_map = density_list[0].astype(float)
        corr_map[corr_map > 0] = 1

    # Slightly cut the bundle at the edgge to clean up single streamline voxels
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
    concat_sft.to_vox()
    concat_sft.to_corner()
    for i in range(len(sft_list)):
        sft_list[i] = cut_outside_of_mask_streamlines(sft_list[i],
                                                      binary_bundle)
        if len(sft_list[i]):
            concat_sft += sft_list[i]
    print('Chop time', time()-t0)
    t0 = time()
    args.nb_pts = len(sft_centroid.streamlines[0]) if args.nb_pts is None \
        else args.nb_pts

    sft_centroid = resample_streamlines_num_points(sft_centroid, args.nb_pts)
    uniformize_bundle_sft(concat_sft, ref_bundle=sft_centroid[0])
    tmp_sft = resample_streamlines_num_points(concat_sft[0:2500], args.nb_pts)

    print('Uni+SLR time', time()-t0)
    t0 = time()
    t0 = time()
    if not args.new_labelling:
        indices = np.array(np.nonzero(binary_bundle), dtype=int).T
        labels = min_dist_to_centroid(indices,
                                      sft_centroid[0].streamlines._data,
                                      nb_pts=args.nb_pts)
        labels_map = np.zeros(binary_bundle.shape, dtype=np.int16)
        labels_map[np.where(binary_bundle)] = labels
        barycenters = compute_euclidean_barycenters(labels_map)
        nib.save(nib.Nifti1Image(labels_map, sft_list[0].affine),
                 os.path.join(args.out_dir, 'labels_map_1.nii.gz'))
        labels = ndi.map_coordinates(labels_map,
                                     concat_sft.streamlines._data.T-0.5,
                                     order=0)
        print('Euclidian time', time()-t0)
        t0 = time()
    else:
        svc = SVC(C=1, kernel='rbf', cache_size=1000)
        labels = transfer_and_diffuse_labels(tmp_sft, sft_centroid)
        print('Diffuse time', time()-t0)
        t0 = time()
        svc.fit(X=tmp_sft.streamlines._data, y=labels)
        print('Fit time', time()-t0)
        t0 = time()
        # print(exp_labels)

        exp_labels = svc.predict(X=np.array(np.where(binary_bundle)).T)
        print('Predict time', time()-t0)
        t0 = time()

        exp_labels_map = np.zeros(binary_bundle.shape, dtype=np.int16)
        exp_labels_map[np.where(binary_bundle)] = exp_labels
        barycenters = compute_shell_barycenters(exp_labels_map)
        exp_labels = svc.predict(X=barycenters)

        labels = ndi.map_coordinates(exp_labels_map,
                                     concat_sft.streamlines._data.T-0.5,
                                     order=0)
        print('Map making time', time()-t0)
        t0 = time()

    barycenter_sft = StatefulTractogram([barycenters], sft_centroid,
                                        space=Space.VOX, origin=Origin.TRACKVIS)

    # It is not allowed that labels jumps labels for consistency
    # Streamlines should have continous labels
    final_streamlines = []
    final_label = []
    curr_ind = 0
    for i, streamline in enumerate(concat_sft.streamlines):
        next_ind = curr_ind + len(streamline)
        curr_labels = labels[curr_ind:next_ind]
        curr_ind = next_ind

        # Flip streamlines so the labels increase (facilitate if/else)
        # Should always be ordered in nextflow pipeline
        gradient = np.gradient(curr_labels)
        if len(np.argwhere(gradient < 0)) > len(np.argwhere(gradient > 0)):
            streamline = streamline[::-1]
            curr_labels = curr_labels[::-1]

        # # Find jumps, cut them and find the longest
        gradient = np.ediff1d(curr_labels)
        max_jump = 2
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

        final_streamlines.append(streamline)
        final_label.append(curr_labels)

    final_streamlines = ArraySequence(final_streamlines)
    final_labels = ArraySequence(final_label)

    indices = np.array(np.nonzero(binary_bundle), dtype=int).T
    labels = min_dist_to_centroid(indices,
                                  final_streamlines._data,
                                  pre_computed_labels=final_labels._data)
    labels_map = np.zeros(binary_bundle.shape, dtype=np.int16)
    labels_map[np.where(binary_bundle)] = labels
    print('Clean Up time', time()-t0)
    t0 = time()
    dists = np.ones(binary_bundle.shape, dtype=float) * -1

    save_tractogram(barycenter_sft, os.path.join(args.out_dir,
                                                    'barycenters.trk'))
                                                    
    import dijkstra3d
    barycenter_bin = compute_tract_counts_map(barycenter_sft.streamlines,
                                                  barycenter_sft.dimensions)
    barycenter_bin[barycenter_bin > 0] = 1
    # for label in range(1, args.nb_pts+1):
    #     indices = np.array(np.nonzero(labels_map == label), dtype=int).T
    #     field = np.ones(labels_map.shape, dtype=float)
    #     for ind in indices:
    #         ind = tuple(ind)
    #         
    #         path = dijkstra3d.dijkstra(field, barycenter, ind, compass=True)
    #         dists[ind] = len(path)-1
    for label in range(1, args.nb_pts+1):
        mask = np.zeros(labels_map.shape)
        mask[labels_map == label] = 1
        barycenter_bin_intersect = barycenter_bin * mask
        barycenter_intersect_coords = np.array(np.nonzero(barycenter_bin_intersect),
                                               dtype=int).T
        bundle_disjoint, num_labels = ndi.label(mask)
        iterations = 0
        
        while num_labels > 1:
            mask = ndi.binary_dilation(mask)
            bundle_disjoint, num_labels = ndi.label(mask)
            iterations += 1
            print('a', label, iterations, num_labels)

        barycenter = tuple(np.round(barycenters[label-1]).astype(int))
        print(label, labels_map[barycenter], barycenter)
        curr_dists = dijkstra3d.distance_field(mask,
                                               source=barycenter_intersect_coords)
        dists[labels_map == label] = curr_dists[labels_map == label]
        print(np.unique(curr_dists, return_counts=True))
        print()

    print('Dijkstra time', time()-t0)
    t0 = time()

    cmap = get_colormap(args.colormap)
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
        nib.save(nib.Nifti1Image(binary_list[i]*corr_map,
                                 sft_list[0].affine),
                 os.path.join(sub_out_dir, 'correlation_map.nii.gz'))
        nib.save(nib.Nifti1Image(binary_list[i]*dists,
                                 sft_list[0].affine),
                 os.path.join(sub_out_dir, 'distance_map.nii.gz'))

        if len(sft):
            tmp_labels = ndi.map_coordinates(labels_map,
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

        if len(sft) and len(args.in_bundles) > 1:
            new_sft.data_per_point['color']._data = cmap(tmp_corr)[
                :, 0:3] * 255
        save_tractogram(new_sft,
                        os.path.join(sub_out_dir, 'correlation.trk'))
    print('Finish time', time()-t0)
    t0 = time()


if __name__ == '__main__':
    main()
