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
  patch-wise correlation map between density maps as a proxy for confidence in
  the bundle's reconstruction.

  The correlation map can be thresholded to remove low confidence regions.
  It will also compute the labels maps for after concatenating bundles,
  ensuring that the labels are spatially consistent between bundles.

# Hyperplane method
  The default is to use the euclidian/centerline method, which is fast and
  works well for most cases.

  The hyperplane method allows for more complex shapes and to split the bundles
  into subsections that follow the geometry of each kind of bundle.
  However, this method is slower and requires extra quality control to ensure
  that the labels are correct. This method requires a centroid file that
  contains multiple streamlines.

  This method is based on the following paper [1], but was heavily modified
  and adapted to work more robustly across datasets.

# Manhattan distance
  The default distance (to barycenter of label) is the euclidian distance.
  The manhattan distance can be used instead to compute the distance to the
  barycenter without stepping out of the mask.

Colormap selection affects tractograms coloring for visualization only.
For detailed information on usage and parameters, please refer to the script's
documentation.

Formerly: scil_compute_bundle_voxel_label_map.py

Author:
-------
Francois Rheault
francois.m.rheault@usherbrooke.ca

------------------------------------------------------------------------------------------
Reference:
[1] Neher, Peter, Dusan Hirjak, and Klaus Maier-Hein. "Radiomic tractometry: a
    rich and tract-specific class of imaging biomarkers for neuroscience and
    medical applications." Research Square (2023).
------------------------------------------------------------------------------------------
"""

import argparse
import logging
import os
import time

from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import is_header_compatible
from dipy.tracking.streamline import transform_streamlines
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
import scipy.ndimage as ndi

from scilpy.image.volume_math import neighborhood_correlation_
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             load_matrix_in_any_format,
                             ranged_type)
from scilpy.tractanalysis.bundle_operations import uniformize_bundle_sft
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.distance_to_centroid import (subdivide_bundles,
                                                       compute_distance_map)
from scilpy.tractograms.streamline_and_mask_operations import \
    cut_streamlines_with_mask, CuttingStyle
from scilpy.tractograms.streamline_operations import \
    filter_streamlines_by_nb_points, remove_overlapping_points_streamlines
from scilpy.viz.color import get_lookup_table
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

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
    p.add_argument('--threshold', type=ranged_type(float, 0, None),
                   default=0.001,
                   help='Maximum distance between two points to be considered '
                        'overlapping [%(default)s mm].')
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
    p.add_argument('--correlation_thr', type=float, const=0.1, nargs='?',
                   default=0,
                   help='Threshold for the correlation map. Only for multi '
                        'bundle case. [%(default)s]')
    p.add_argument('--streamlines_thr', type=int, const=2, nargs='?',
                   help='Threshold for the minimum number of streamlines in a '
                        'voxel to be included [%(default)s].')
    p.add_argument('--transformation',
                   help='Transformation matrix to apply to the centroid')
    p.add_argument('--inverse', action='store_true',
                   help='Inverse the transformation matrix.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(args.verbose)

    assert_inputs_exist(parser, args.in_bundles + [args.in_centroid],
                        optional=args.reference)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)

    if args.streamlines_thr is not None and args.streamlines_thr < 1:
        parser.error('streamlines_thr must be greater than 1.')

    if args.correlation_thr < 0:
        parser.error('correlation_thr must be greater than 0')

    if args.hyperplane:
        warning = 'WARNING: The --hyperplane option should be used carefully,'\
                  ' not fully tested!'
        heading = '=' * len(warning)
        logging.warning(f'{heading}\n{warning}\n{heading}')

    sft_centroid = load_tractogram_with_reference(parser, args,
                                                  args.in_centroid)
    if args.transformation is not None:
        streamlines = sft_centroid.streamlines
        transfo = load_matrix_in_any_format(args.transformation)
        if args.inverse:
            transfo = np.linalg.inv(transfo)
        streamlines = transform_streamlines(
            streamlines, transfo, in_place=False)
        sft_centroid = StatefulTractogram(streamlines, args.in_bundles[0],
                                          space=Space.RASMM)

    # When doing longitudinal data, the split in subsection can be done
    # on all the bundles at once. Needs to be co-registered.
    timer = time.time()
    sft_list = []
    for filename in args.in_bundles:
        sft = load_tractogram_with_reference(parser, args, filename)
        if not len(sft.streamlines):
            raise IOError(f'Empty bundle file {args.in_bundles}. Skipping.')
        if not args.skip_uniformize:
            uniformize_bundle_sft(sft, ref_bundle=sft_centroid)
        sft.to_vox()
        sft.to_corner()

        # Only process sft containing more than 1 streamlines
        if len(sft.streamlines) > 1:
            sft_list.append(sft)
        else:
            logging.warning("Bundle {} contains less than 2 streamlines."
                            " It won't be processed.".format(filename))

        if len(sft_list):
            if not is_header_compatible(sft_list[0], sft_list[-1]):
                parser.error(f'Header of {args.in_bundles[0]} and '
                             f'{filename} are not compatible')

    sft_centroid.to_vox()
    sft_centroid.to_corner()
    logging.debug(f'Loaded {len(args.in_bundles)} bundle(s) in '
                  f'{round(time.time() - timer, 3)} seconds.')

    if len(sft_list) == 0:
        logging.error('No bundle to process. Exiting.')
        return

    density_list = []
    binary_list = []
    timer = time.time()
    for sft in sft_list:
        density = compute_tract_counts_map(sft.streamlines,
                                           sft.dimensions).astype(float)
        binary = np.zeros(sft.dimensions, dtype=np.uint8)
        if args.streamlines_thr is not None:
            binary[density >= args.streamlines_thr] = 1
        else:
            binary[density > 0] = 1
        binary_list.append(binary)
        density_list.append(density)

    if not is_header_compatible(sft_centroid, sft_list[0]):
        raise IOError(f'{args.in_centroid} and {args.in_bundles} do not have a'
                      ' compatible header')

    logging.debug('Computed density and binary map(s) in '
                  f'{round(time.time() - timer, 3)}.')

    if len(density_list) > 1:
        timer = time.time()
        corr_map = neighborhood_correlation_(density_list)
        logging.debug(f'Computed correlation map in '
                      f'{round(time.time() - timer, 3)} seconds')
    else:
        corr_map = density_list[0].astype(float)
        corr_map[corr_map > 0] = 1

    # Slightly cut the bundle at the edge to clean up single streamline voxels
    # with no neighbor. Remove isolated voxels to keep a single 'blob'
    binary_mask = np.max(binary_list, axis=0)

    if args.correlation_thr > 1e-3:
        binary_mask[corr_map < args.correlation_thr] = 0

    # A bundle must be contiguous, we can't have isolated voxels.
    # We will cut it later
    if args.streamlines_thr is not None:
        bundle_disjoint, _ = ndi.label(binary_mask)
        unique, count = np.unique(bundle_disjoint, return_counts=True)
        val = unique[np.argmax(count[1:])+1]
        binary_mask[bundle_disjoint != val] = 0

    nib.save(nib.Nifti1Image(corr_map * binary_mask, sft_list[0].affine),
             os.path.join(args.out_dir, 'correlation_map.nii.gz'))

    # Trim the bundle(s), remove voxels with poor correlation or
    # isolated components.
    timer = time.time()
    concat_sft = StatefulTractogram.from_sft([], sft_list[0])
    concat_sft.to_vox()
    concat_sft.to_corner()
    for i in range(len(sft_list)):
        sft_list[i] = cut_streamlines_with_mask(sft_list[i],
                                                binary_mask)

        sft_list[i] = filter_streamlines_by_nb_points(sft_list[i],
                                                      min_nb_points=4)
        if len(sft_list[i]):
            concat_sft += sft_list[i]

    logging.debug(
        f'Trim bundle(s) in {round(time.time() - timer, 3)} seconds.')

    method = 'hyperplane' if args.hyperplane else 'centerline'
    args.nb_pts = len(sft_centroid.streamlines[0]) if args.nb_pts is None \
        else args.nb_pts
    labels_map = subdivide_bundles(concat_sft, sft_centroid, binary_mask,
                                   args.nb_pts, method=method)

    # We trim the streamlines due to looping labels, so we have a new binary
    # mask
    binary_mask = np.zeros(labels_map.shape, dtype=np.uint8)
    binary_mask[labels_map > 0] = 1

    # We need to count blobs again, as the labels could be not contiguous
    labelized, count = ndi.label(binary_mask)
    unique, count = np.unique(labelized, return_counts=True)
    ratio = count[1] / np.sum(count[1:])

    # 0.9 is arbitrary, but we want a vast majority of the voxels to be
    # contiguous, otherwise it is a weird bundle so we recompute the labels
    # using the centerline method.
    if len(unique) > 2 and ratio < 0.9:
        binary_mask = np.max(binary_list, axis=0)
        labels_map = subdivide_bundles(concat_sft, sft_centroid, binary_mask,
                                       args.nb_pts, method='centerline',
                                       fix_jumps=False)
        logging.warning('Warning: Some labels were not contiguous. '
                        'Recomputing labels to centerline method.')

    timer = time.time()
    distance_map = compute_distance_map(labels_map, binary_mask, args.nb_pts,
                                        use_manhattan=args.use_manhattan)
    logging.debug('Computed distance map in '
                  f'{round(time.time() - timer, 3)} seconds')

    timer = time.time()
    cmap = get_lookup_table(args.colormap)
    for i, sft in enumerate(sft_list):
        if len(sft_list) > 1:
            sub_out_dir = os.path.join(args.out_dir, f'session_{i+1}')
        else:
            sub_out_dir = args.out_dir
        timer = time.time()
        new_sft = StatefulTractogram.from_sft(sft.streamlines, sft_list[0])
        cut_sft = cut_streamlines_with_mask(
            new_sft, binary_mask,
            cutting_style=CuttingStyle.KEEP_LONGEST)

        cut_sft = remove_overlapping_points_streamlines(cut_sft,
                                                        args.threshold)
        cut_sft = filter_streamlines_by_nb_points(cut_sft, min_nb_points=2)

        logging.debug(
            f'Cut streamlines in {round(time.time() - timer, 3)} seconds')
        cut_sft.data_per_point['color'] = ArraySequence(cut_sft.streamlines)
        if not os.path.isdir(sub_out_dir):
            os.mkdir(sub_out_dir)

        # Dictionary to hold the data for each type
        data_dict = {'labels': labels_map.astype(np.uint16),
                     'distance': distance_map.astype(np.float32),
                     'correlation': corr_map.astype(np.float32)}

        # Iterate through each type to save the files
        for basename, map in data_dict.items():
            nib.save(
                nib.Nifti1Image((binary_list[i] * map), sft_list[0].affine),
                os.path.join(sub_out_dir, f'{basename}_map.nii.gz'))

            if basename == 'correlation' and len(args.in_bundles) == 1:
                continue

            if len(cut_sft):
                tmp_data = ndi.map_coordinates(
                    map, cut_sft.streamlines._data.T - 0.5, order=0)

                if basename == 'labels':
                    max_val = args.nb_pts
                elif basename == 'correlation':
                    max_val = 1
                else:
                    max_val = np.max(tmp_data)
                max_val = args.nb_pts
                cut_sft.data_per_point['color']._data = cmap(
                    tmp_data / max_val)[:, 0:3] * 255

                # Save the tractogram
                save_tractogram(cut_sft,
                                os.path.join(sub_out_dir,
                                             f'{basename}.trk'))


if __name__ == '__main__':
    main()
