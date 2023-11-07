#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute label image (Nifti) from bundle and centroid.
Each voxel will have the label of its nearest centroid point.

The number of labels will be the same as the centroid's number of points.
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import RBFSampler
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

from scilpy.image.labels import weighted_vote_median_filter
from scilpy.image.volume_math import correlation
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.distance_to_centroid import (min_dist_to_centroid,
                                                       compute_distance_map,
                                                       associate_labels)
from scilpy.tractograms.streamline_and_mask_operations import \
    cut_outside_of_mask_streamlines
from scilpy.tractograms.streamline_operations import \
    resample_streamlines_num_points, resample_streamlines_step_size
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
    p.add_argument('--skip_uniformize', action='store_true',
                   help='Skip uniformization of the bundles orientation.')

    add_processes_arg(p)
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

    sft_list = []
    for filename in args.in_bundles:
        sft = load_tractogram_with_reference(parser, args, filename)
        if not len(sft.streamlines):
            raise IOError('Empty bundle file {}. '
                          'Skipping'.format(args.in_bundle))
        if not args.skip_uniformize:
            uniformize_bundle_sft(sft, ref_bundle=sft_centroid)
        sft.to_vox()
        sft.to_corner()
        sft_list.append(sft)

        if len(sft_list):
            if not is_header_compatible(sft_list[0], sft_list[-1]):
                parser.error('Header of {} and {} are not compatible'.format(
                    args.in_bundles[0], filename))

    # Perform after the uniformization
    sft_centroid.to_vox()
    sft_centroid.to_corner()

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
    binary_map = np.zeros(corr_map.shape, dtype=bool)
    binary_map[corr_map > 0.5] = 1

    bundle_disjoint, _ = ndi.label(binary_map)
    unique, count = np.unique(bundle_disjoint, return_counts=True)
    val = unique[np.argmax(count[1:])+1]
    binary_map[bundle_disjoint != val] = 0

    corr_map = corr_map*binary_map
    nib.save(nib.Nifti1Image(corr_map, sft_list[0].affine),
             os.path.join(args.out_dir, 'correlation_map.nii.gz'))

    # Chop off some streamlines
    concat_sft = StatefulTractogram.from_sft([], sft_list[0])
    concat_sft.to_vox()
    concat_sft.to_corner()
    for i in range(len(sft_list)):
        sft_list[i] = cut_outside_of_mask_streamlines(sft_list[i],
                                                      binary_map)
        if len(sft_list[i]):
            concat_sft += sft_list[i]

    t0 = time()
    args.nb_pts = len(sft_centroid.streamlines[0]) if args.nb_pts is None \
        else args.nb_pts

    sft_centroid = resample_streamlines_num_points(sft_centroid, args.nb_pts)

    # Select 2000 elements from the SFTs
    random_indices = np.random.choice(len(concat_sft),
                                      min(len(concat_sft), 2000),
                                      replace=False)
    tmp_sft = resample_streamlines_step_size(concat_sft[random_indices], 2.0)

    t0 = time()
    if not args.new_labelling:
        indices = np.array(np.nonzero(binary_map), dtype=int).T
        labels = min_dist_to_centroid(indices,
                                      sft_centroid[0].streamlines._data,
                                      nb_pts=args.nb_pts)
    else:
        labels, _, _ = associate_labels(tmp_sft, sft_centroid,
                                        args.nb_pts)

        # Initialize the scaler and the RBF sampler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        rbf_feature = RBFSampler(gamma=1.0, n_components=500, random_state=1)

        # Fit the scaler to the streamline data and transform it
        scaler.fit(tmp_sft.streamlines._data)
        scaled_streamline_data = scaler.transform(tmp_sft.streamlines._data)

        # Fit the RBFSampler to the scaled data and transform it
        rbf_feature.fit(scaled_streamline_data)
        features = rbf_feature.transform(scaled_streamline_data)

        # Initialize and fit the SGDClassifier with log loss
        sgd_clf = SGDClassifier(loss='log_loss', max_iter=10000, tol=1e-4,
                                alpha=0.0001, random_state=1,
                                n_jobs=min(args.nb_pts, args.nbr_processes))
        sgd_clf.fit(X=features, y=labels)

        # Scale the coordinates of the voxels and transform with RBFSampler
        voxel_coords = np.array(np.where(binary_map)).T
        scaled_voxel_coords = scaler.transform(voxel_coords)
        transformed_voxel_coords = rbf_feature.transform(scaled_voxel_coords)

        # Predict the labels for the voxels
        labels = sgd_clf.predict(X=transformed_voxel_coords)

    print('SVC time for {}'.format(args.out_dir), time()-t0)

    labels_map = np.zeros(binary_map.shape, dtype=np.uint16)
    labels_map[np.where(binary_map)] = labels
    distance_map = compute_distance_map(labels_map, binary_map,
                                        args.new_labelling, args.nb_pts)

    cmap = get_colormap(args.colormap)
    for i, sft in enumerate(sft_list):
        if len(sft_list) > 1:
            sub_out_dir = os.path.join(args.out_dir, 'session_{}'.format(i+1))
        else:
            sub_out_dir = args.out_dir

        new_sft = StatefulTractogram.from_sft(sft.streamlines, sft_list[0])
        new_sft.data_per_point['color'] = ArraySequence(new_sft.streamlines)
        if not os.path.isdir(sub_out_dir):
            os.mkdir(sub_out_dir)

        # Dictionary to hold the data for each type
        data_dict = {'labels': labels_map.astype(np.uint16),
                     'distance': distance_map.astype(float),
                     'correlation': corr_map.astype(float)}

        # Iterate through each type to save the files
        for basename, map in data_dict.items():
            nib.save(nib.Nifti1Image((binary_list[i] * map), sft_list[0].affine),
                     os.path.join(sub_out_dir, "{}_map.nii.gz".format(basename)))

            if basename == 'correlation' and len(args.in_bundles) == 1:
                continue

            if len(sft):
                tmp_data = ndi.map_coordinates(
                    map, sft.streamlines._data.T - 0.5, order=0)

                max_val = args.nb_pts if basename == 'labels' else np.max(
                    tmp_data)
                new_sft.data_per_point['color']._data = cmap(
                    tmp_data / max_val)[:, 0:3] * 255

                # Save the tractogram
                save_tractogram(new_sft,
                                os.path.join(sub_out_dir,
                                             "{}.trk".format(basename)))


if __name__ == '__main__':
    main()
