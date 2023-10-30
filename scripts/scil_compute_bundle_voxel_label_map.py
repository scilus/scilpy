#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute label image (Nifti) from bundle and centroid.
Each voxel will have the label of its nearest centroid point.

The number of labels will be the same as the centroid's number of points.
"""

import argparse
import os

from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, set_sft_logger_level, Space
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
from scilpy.tractanalysis.distance_to_centroid import min_dist_to_centroid
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

from sklearn.svm import SVC
from time import time

from collections import defaultdict

def compute_overlap_and_mapping(small_data, full_data):
    """
    Compute the overlap between labels in the small and full datasets
    and generate a mapping based on maximizing this overlap.
    
    Parameters:
    - small_data: np.ndarray, data from the smaller image
    - full_data: np.ndarray, data from the full image
    - unique_small: np.ndarray, unique labels in the smaller image
    - unique_full: np.ndarray, unique labels in the full image
    
    Returns:
    - dict, mapping from labels in the smaller image to labels in the full image
    """
    mapping = {}
    unique_small = np.unique(small_data)
    unique_full = np.unique(full_data)
    for label_small in unique_small:
        if label_small == 0:
            continue  # Skip background
        
        overlaps = defaultdict(int)
        mask_small = small_data == label_small

        for label_full in unique_full:
            if label_full == 0:
                continue  # Skip background
            
            mask_full = full_data == label_full
            overlap = np.sum(mask_small & mask_full)
            
            if overlap > 0:
                overlaps[label_full] = overlap

        if overlaps:
            best_match = max(overlaps, key=overlaps.get)
            mapping[label_small] = best_match
        else:
            # If no overlap found, continue with default increasing +1 scheme
            mapping[label_small] = label_small + 1
    
    return mapping

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

    args.nb_pts = len(sft_centroid.streamlines[0]) if args.nb_pts is None \
        else args.nb_pts

    sft_centroid = resample_streamlines_num_points(sft_centroid, args.nb_pts)
    uniformize_bundle_sft(concat_sft, ref_bundle=sft_centroid[0])
    tmp_sft = resample_streamlines_num_points(concat_sft[0:1000], args.nb_pts)

    if not args.new_labelling:
        new_streamlines = sft_centroid.streamlines.copy()
        sft_centroid = StatefulTractogram.from_sft([new_streamlines[0]],
                                                   sft_centroid)
    else:
        srr = StreamlineLinearRegistration()
        srm = srr.optimize(static=tmp_sft.streamlines,
                        moving=sft_centroid.streamlines)
        sft_centroid.streamlines = srm.transform(sft_centroid.streamlines)

    t0 = time()
    if not args.new_labelling:
        labels, _ = min_dist_to_centroid(concat_sft.streamlines._data,
                                            sft_centroid.streamlines._data,
                                            args.nb_pts)
        labels += 1  # 0 means no labels
        labels_map = np.zeros(binary_bundle.shape, dtype=np.int16)
        indices = np.array(np.nonzero(binary_bundle), dtype=int).T

        kd_tree = cKDTree(concat_sft.streamlines._data)
        for ind in indices:
            _, neighbor_ids = kd_tree.query(ind, k=5)

            if not len(neighbor_ids):
                continue

            labels_val = labels[neighbor_ids]

            vote = np.bincount(labels_val)
            total = np.arange(np.amax(labels_val+1))
            winner = total[np.argmax(vote)]
            labels_map[ind[0], ind[1], ind[2]] = winner

    else:
        svc = SVC(C=1, kernel='rbf')
        labels = np.tile(np.arange(0,args.nb_pts)[::-1], len(sft_centroid))
        labels += 1
        svc.fit(X=sft_centroid.streamlines._data, y=labels)

        labels_pred = svc.predict(X=np.array(np.where(binary_bundle)).T)
        labels_map = np.zeros(binary_bundle.shape, dtype=np.int16)
        labels_map[np.where(binary_bundle)] = labels_pred

    print('a', time()-t0)



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
        # nib.save(nib.Nifti1Image(binary_list[i]*distance_map,
        #                          sft_list[0].affine),
        #          os.path.join(sub_out_dir, 'distance_map.nii.gz'))
        nib.save(nib.Nifti1Image(binary_list[i]*corr_map,
                                 sft_list[0].affine),
                 os.path.join(sub_out_dir, 'correlation_map.nii.gz'))

        if len(sft):
            tmp_labels = ndi.map_coordinates(labels_map,
                                             sft.streamlines._data.T-0.5,
                                             order=0)
            # tmp_dists = ndi.map_coordinates(distance_map,
            #                                 sft.streamlines._data.T-0.5,
            #                                 order=0)
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

        # if len(sft):
        #     new_sft.data_per_point['color']._data = cmap(
        #         tmp_dists / np.max(tmp_dists))[:, 0:3] * 255
        # save_tractogram(new_sft,
        #                 os.path.join(sub_out_dir, 'distance.trk'))

        if len(sft):
            new_sft.data_per_point['color']._data = cmap(tmp_corr)[
                :, 0:3] * 255
        save_tractogram(new_sft,
                        os.path.join(sub_out_dir, 'correlation.trk'))


if __name__ == '__main__':
    main()
