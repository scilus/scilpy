#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute label image (Nifti) from bundle(s) and centroid(s).
Each voxel will have a label that represents its position along the bundle.

The number of labels will be the same as the centroid's number of points,
unless specified otherwise.

# Single bundle case
  This script takes as input a bundle file, a centroid streamline corresponding
  to the bundle. It computes label images, where each voxel is assigned the
  label of its nearest centroid point. The resulting images represent the
  labels, distances between the bundle and centroid.

# Multiple bundle case
  When providing multiple (co-registered) bundles, the script will compute a
  correlation map, which shows the spatial correlation between density maps
  It will also compute the labels maps for all bundles at once, ensuring
  that the labels are spatial consistent between bundles.

# Hyperplane method
  The default is to use the euclidian/centerline method, which is fast and
  works well for most cases.
  The hyperplane method allows for more complex shapes and to split the bundles
  into subsection that follow the geometry of each kind of bundle.
  However, this method is slower and requires extra quality control to ensure
  that the labels are correct. This method requires a centroid file that
  contains multiple streamlines.
  This method is based on the following paper [1], but was heavily modified
  and adapted to work more robustly across datasets.

# Manhatan distance
  The default distance (to barycenter of label) is the euclidian distance.
  The manhattan distance can be used instead to compute the distance to the
  barycenter without stepping out of the mask.

Colormap selection affects tractograms coloring for visualization only.
For detailed information on usage and parameters, please refer to the script's
documentation.

Author:
-------
Francois Rheault
francois.m.rheault@usherbrooke.ca
"""

import argparse
import logging
import os
import time

from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.utils import is_header_compatible
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
import scipy.ndimage as ndi
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler

from scilpy.image.volume_math import correlation
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.distance_to_centroid import (min_dist_to_centroid,
                                                       compute_distance_map,
                                                       associate_labels,
                                                       correct_labels_jump)
from scilpy.tractograms.streamline_and_mask_operations import \
    cut_outside_of_mask_streamlines
from scilpy.tractograms.streamline_operations import \
    resample_streamlines_num_points, resample_streamlines_step_size
from scilpy.utils.streamlines import uniformize_bundle_sft
from scilpy.viz.utils import get_colormap


EPILOG = """
[1] Neher, Peter, Dusan Hirjak, and Klaus Maier-Hein. "Radiomic tractometry: a
    rich and tract-specific class of imaging biomarkers for neuroscience and
    medical applications." Research Square (2023).
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=EPILOG,
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
    p.add_argument('--hyperplane', action='store_true',
                   help='Use the hyperplane method (multi-centroids) instead '
                        'of the euclidian method (single-centroid).')
    p.add_argument('--use_manhattan', action='store_true',
                   help='Use the manhattan distance instead of the euclidian '
                   'distance.')
    p.add_argument('--skip_uniformize', action='store_true',
                   help='Skip uniformization of the bundles orientation.')
    p.add_argument('--correlation_thr', type=float, default=0.1,
                   help='Threshold for the correlation map. Only for multi '
                        'bundle case. [%(default)s]')

    add_processes_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    # set_sft_logger_level('ERROR')
    assert_inputs_exist(parser, args.in_bundles + [args.in_centroid],
                        optional=args.reference)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)

    sft_centroid = load_tractogram_with_reference(parser, args,
                                                  args.in_centroid)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # When doing longitudinal data, the split in subsection can be done
    # on all the bundles at once. Needs to be co-registered.
    timer = time.time()
    sft_list = []
    for filename in args.in_bundles:
        sft = load_tractogram_with_reference(parser, args, filename)
        if not len(sft.streamlines):
            raise IOError('Empty bundle file {}. '
                          'Skipping'.format(args.in_bundles))
        if not args.skip_uniformize:
            uniformize_bundle_sft(sft, ref_bundle=sft_centroid)
        sft.to_vox()
        sft.to_corner()
        sft_list.append(sft)

        if len(sft_list):
            if not is_header_compatible(sft_list[0], sft_list[-1]):
                parser.error('Header of {} and {} are not compatible'.format(
                    args.in_bundles[0], filename))

    sft_centroid.to_vox()
    sft_centroid.to_corner()
    logging.info('Loaded {} bundle(s) in {} seconds.'.format(
        len(args.in_bundles), round(time.time() - timer, 3)))

    density_list = []
    binary_list = []
    timer = time.time()
    for sft in sft_list:
        density = compute_tract_counts_map(sft.streamlines,
                                           sft.dimensions).astype(float)
        binary = np.zeros(sft.dimensions, dtype=np.uint8)
        binary[density > 0] = 1
        binary_list.append(binary)
        density_list.append(density)

    if not is_header_compatible(sft_centroid, sft_list[0]):
        raise IOError('{} and {}do not have a compatible header'.format(
            args.in_centroid, args.in_bundles))
    logging.info('Computed density and binary map(s) in {}.'.format(
        round(time.time() - timer, 3)))

    if len(density_list) > 1:
        timer = time.time()
        corr_map = correlation(density_list, None)
        logging.info('Computed correlation map in {} seconds.'.format(
            round(time.time() - timer, 3)))
    else:
        corr_map = density_list[0].astype(float)
        corr_map[corr_map > 0] = 1

    # Slightly cut the bundle at the edgge to clean up single streamline voxels
    # with no neighbor. Remove isolated voxels to keep a single 'blob'
    binary_map = np.zeros(corr_map.shape, dtype=bool)
    binary_map[corr_map > args.correlation_thr] = 1

    bundle_disjoint, _ = ndi.label(binary_map)
    unique, count = np.unique(bundle_disjoint, return_counts=True)
    val = unique[np.argmax(count[1:])+1]
    binary_map[bundle_disjoint != val] = 0

    nib.save(nib.Nifti1Image(corr_map * binary_map, sft_list[0].affine),
             os.path.join(args.out_dir, 'correlation_map.nii.gz'))

    # A bundle must be contiguous, we can't have isolated voxels.
    timer = time.time()
    concat_sft = StatefulTractogram.from_sft([], sft_list[0])
    concat_sft.to_vox()
    concat_sft.to_corner()
    for i in range(len(sft_list)):
        sft_list[i] = cut_outside_of_mask_streamlines(sft_list[i],
                                                      binary_map)
        if len(sft_list[i]):
            concat_sft += sft_list[i]
    logging.info('Chop bundle(s) in {} seconds.'.format(
        round(time.time() - timer, 3)))

    args.nb_pts = len(sft_centroid.streamlines[0]) if args.nb_pts is None \
        else args.nb_pts

    # This allows to have a more uniform (in size) first and last labels
    if args.hyperplane:
        args.nb_pts += 2

    sft_centroid = resample_streamlines_num_points(sft_centroid, args.nb_pts)

    timer = time.time()
    if not args.hyperplane:
        indices = np.array(np.nonzero(binary_map), dtype=int).T
        labels = min_dist_to_centroid(indices,
                                      sft_centroid[0].streamlines._data,
                                      nb_pts=args.nb_pts)
        logging.info('Computed labels using the euclidian method '
                     'in {} seconds'.format(round(time.time() - timer, 3)))
    else:
        logging.info('Computing Labels using the hyperplane method.\n'
                     '\tThis can take a while...')
        # Select 2000 elements from the SFTs to train the classifier
        random_indices = np.random.choice(len(concat_sft),
                                          min(len(concat_sft), 2000),
                                          replace=False)
        tmp_sft = resample_streamlines_step_size(concat_sft[random_indices],
                                                 1.0)
        # Associate the labels to the streamlines using the centroids as
        # reference (to handle shorter bundles due to missing data)
        mini_timer = time.time()
        labels, _, _ = associate_labels(tmp_sft, sft_centroid,
                                        args.nb_pts)
        print('\tAssociated labels to_centroids in {} seconds'.format(
            round(time.time() - mini_timer, 3)))

        # Initialize the scaler and the RBF sampler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        rbf_feature = RBFSampler(gamma=1.0, n_components=1000, random_state=1)

        # Fit the scaler to the streamline data and transform it
        mini_timer = time.time()
        scaler.fit(tmp_sft.streamlines._data)
        scaled_streamline_data = scaler.transform(tmp_sft.streamlines._data)

        # Fit the RBFSampler to the scaled data and transform it
        rbf_feature.fit(scaled_streamline_data)
        features = rbf_feature.transform(scaled_streamline_data)
        print('\tScaler and RBF kernel approximation in {} seconds'.format(
            round(time.time() - mini_timer, 3)))

        # Initialize and fit the SGDClassifier with log loss
        mini_timer = time.time()
        sgd_clf = SGDClassifier(loss='log_loss', max_iter=10000, tol=1e-5,
                                alpha=0.0001, random_state=1,
                                n_jobs=min(args.nb_pts, args.nbr_processes))
        sgd_clf.fit(X=features, y=labels)
        print('\tSGDClassifier fit of training data in {} seconds'.format(
            round(time.time() - mini_timer, 3)))

        # Scale the coordinates of the voxels and transform with RBFSampler
        mini_timer = time.time()
        voxel_coords = np.array(np.where(binary_map)).T
        scaled_voxel_coords = scaler.transform(voxel_coords)
        transformed_voxel_coords = rbf_feature.transform(scaled_voxel_coords)

        # Predict the labels for the voxels
        labels = sgd_clf.predict(X=transformed_voxel_coords)
        print('\tSGDClassifier prediction of labels in {} seconds'.format(
            round(time.time() - mini_timer, 3)))

        logging.info('Computed labels using the hyperplane method '
                     'in {} seconds'.format(round(time.time() - timer, 3)))
    labels_map = np.zeros(binary_map.shape, dtype=np.uint16)
    labels_map[np.where(binary_map)] = labels

    # Correct the hyperplane labels to have a more uniform size
    if args.hyperplane:
        labels_map[labels_map == args.nb_pts] = args.nb_pts - 1
        labels_map[labels_map == 1] = 2
        labels_map[labels_map > 0] -= 1
        args.nb_pts -= 2

    timer = time.time()
    labels_map = correct_labels_jump(labels_map, concat_sft.streamlines,
                                     args.nb_pts)
    logging.info('Corrected labels jump in {} seconds'.format(
        round(time.time() - timer, 3)))

    timer = time.time()
    distance_map = compute_distance_map(labels_map, binary_map,
                                        args.use_manhattan, args.nb_pts)
    logging.info('Computed distance map in {} seconds'.format(
        round(time.time() - timer, 3)))

    timer = time.time()
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
                     'distance': distance_map.astype(np.float32),
                     'correlation': corr_map.astype(np.float32)}

        # Iterate through each type to save the files
        for basename, map in data_dict.items():
            nib.save(nib.Nifti1Image((binary_list[i] * map), sft_list[0].affine),
                     os.path.join(sub_out_dir, "{}_map.nii.gz".format(basename)))

            if basename == 'correlation' and len(args.in_bundles) == 1:
                continue

            if len(sft):
                tmp_data = ndi.map_coordinates(
                    map, sft.streamlines._data.T - 0.5, order=0)

                if basename == 'labels':
                    max_val = args.nb_pts
                elif basename == 'correlation':
                    max_val = 1
                else:
                    max_val = np.max(tmp_data)
                max_val = args.nb_pts
                new_sft.data_per_point['color']._data = cmap(
                    tmp_data / max_val)[:, 0:3] * 255

                # Save the tractogram
                save_tractogram(new_sft,
                                os.path.join(sub_out_dir,
                                             "{}.trk".format(basename)))
    logging.info('Saved all data to {} in {} seconds'.format(
        args.out_dir, round(time.time() - timer, 3)))


if __name__ == '__main__':
    main()
