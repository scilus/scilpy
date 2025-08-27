#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute a label image (Nifti) from bundle(s) and centroid(s).
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
from dipy.tracking.streamline import transform_streamlines
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
import scipy.ndimage as ndi

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             load_matrix_in_any_format,
                             ranged_type, assert_headers_compatible)
from scilpy.tractanalysis.bundle_operations import uniformize_bundle_sft, \
    keep_main_blob_from_bundle_map
from scilpy.tractanalysis.multi_bundle_operations import get_correlation_map
from scilpy.tractanalysis.distance_to_centroid import (
    compute_distance_map, subdivide_bundle_with_quality_check)
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
                   help='Centroid streamline corresponding to the bundle.')
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
    p.add_argument('--correlation_thr',
                   type=ranged_type(float, min_value=0, min_excluded=False),
                   const=0.1, nargs='?', default=0,
                   help='Threshold for the correlation map. Only for multi '
                        'bundle case. [%(default)s]')
    p.add_argument('--streamlines_thr',
                   type=ranged_type(int, min_value=1, min_excluded=False),
                   const=2, nargs='?',
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


def save_output(sub_out_dir, basename, bmap, cut_sft, ref_affine, bundle_mask,
                cmap, max_val=None, save_trk=True):
    """Save an output file and its associated tractogram."""

    # Save the main file.
    nib.save(nib.Nifti1Image((bundle_mask * bmap), ref_affine),
             os.path.join(sub_out_dir, f'{basename}_map.nii.gz'))

    # Save its associated tractogram
    if save_trk and len(cut_sft) > 0:
        # Add the given data as color
        tmp_data = ndi.map_coordinates(
            bmap, cut_sft.streamlines._data.T - 0.5, order=0,
            mode='nearest')

        if max_val is None:
            max_val = np.max(tmp_data)

        cut_sft.data_per_point['color']._data = cmap(
            tmp_data / max_val)[:, 0:3] * 255

        # Save the tractogram
        save_tractogram(cut_sft,
                        os.path.join(sub_out_dir, f'{basename}.trk'))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(args.verbose)

    # Verifications
    assert_inputs_exist(parser, args.in_bundles + [args.in_centroid],
                        optional=args.reference)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)
    assert_headers_compatible(parser, args.in_centroid, args.in_bundles)

    if args.hyperplane:
        warning = 'WARNING: The --hyperplane option should be used carefully,'\
                  ' not fully tested!'
        heading = '=' * len(warning)
        logging.warning(f'{heading}\n{warning}\n{heading}')

    # Loading the centroid. Other bundles will be loaded in the loop.
    sft_centroid = load_tractogram_with_reference(parser, args,
                                                  args.in_centroid)
    if args.transformation is not None:
        # Transforming the centroid streamlines now.
        streamlines = sft_centroid.streamlines
        transfo = load_matrix_in_any_format(args.transformation)
        if args.inverse:
            transfo = np.linalg.inv(transfo)
        streamlines = transform_streamlines(
            streamlines, transfo, in_place=False)
        sft_centroid = StatefulTractogram(streamlines, args.in_bundles[0],
                                          space=Space.RASMM)
    sft_centroid.to_vox()
    sft_centroid.to_corner()

    # Loading all bundles into sft_list + uniformizing
    timer = time.time()
    sft_list = []
    for filename in args.in_bundles:
        sft = load_tractogram_with_reference(parser, args, filename)
        # Only process sft containing more than 1 streamlines
        if len(sft.streamlines) < 2:
            logging.warning("Bundle {} contains less than 2 streamlines."
                            " It won't be processed.".format(filename))
            continue
        if not args.skip_uniformize:
            uniformize_bundle_sft(sft, ref_bundle=sft_centroid)
        sft.to_vox()
        sft.to_corner()
        sft_list.append(sft)

    if len(sft_list) == 0:
        logging.error('No bundle to process. Exiting.')
        return
    logging.debug(f'Loaded {len(args.in_bundles)} bundle(s) in '
                  f'{round(time.time() - timer, 3)} seconds.')

    # --- Main processing starts ---

    # Prepare the binary mask and correlation map with all bundles
    # Will also filter out voxels with low density.
    corr_map, binary_mask, binary_mask_nothresh, binary_list = (
        get_correlation_map(
            sft_list, args.streamlines_thr, args.correlation_thr))

    # Make sure we didn't end up with separated blobs in the map.
    # Here we don't check the is_ok variable.
    if args.streamlines_thr is not None or args.corralation_thr > 0:
        binary_mask, is_ok, _ = keep_main_blob_from_bundle_map(binary_mask)

        if not is_ok:
            logging.warning("After thresholding, --streamlines_thr, "
                            "--correlation_thr), the final bundle mask seems "
                            "broken into small blobs. Verify the results.")

    # Save the correlation map
    nib.save(nib.Nifti1Image(corr_map * binary_mask, sft_list[0].affine),
             os.path.join(args.out_dir, 'correlation_map.nii.gz'))

    # Cut the bundles inside the mask, remove voxels with poor correlation or
    # isolated components.
    timer = time.time()
    concat_sft = StatefulTractogram.from_sft([], sft_list[0])
    concat_sft.to_vox()
    concat_sft.to_corner()
    for i in range(len(sft_list)):
        sft_list[i] = cut_streamlines_with_mask(sft_list[i], binary_mask)

        sft_list[i] = filter_streamlines_by_nb_points(sft_list[i],
                                                      min_nb_points=4)
        if len(sft_list[i]):
            concat_sft += sft_list[i]
    logging.debug(
        f'Trim bundle(s) in {round(time.time() - timer, 3)} seconds.')

    # Subdivision into labels
    # When doing longitudinal data, subdivision into sections can be done on
    # all the bundles at once if they are co-registered.
    method = 'hyperplane' if args.hyperplane else 'centerline'
    args.nb_pts = len(sft_centroid.streamlines[0]) if args.nb_pts is None \
        else args.nb_pts
    labels_map = subdivide_bundle_with_quality_check(
        concat_sft, sft_centroid, binary_mask, method, args.nb_pts,
        alternate_mask=binary_mask_nothresh)

    # Computing distance map
    timer = time.time()
    distance_map = compute_distance_map(labels_map, binary_mask, args.nb_pts,
                                        use_manhattan=args.use_manhattan)
    logging.debug('Computed distance map in '
                  f'{round(time.time() - timer, 3)} seconds')

    # Save each bundle
    # - Keep final bundles inside the masks
    # - Slightly cut the bundle at the edge to clean up single streamline
    #   voxels with no neighbor.
    # - remove_overlapping_points_streamlines. WHY?
    # - keep only streamlines with at least 2 pts
    cmap = get_lookup_table(args.colormap)
    ref_affine = sft_list[0].affine
    for i, sft in enumerate(sft_list):

        # Decide the output directory
        if len(sft_list) > 1:
            sub_out_dir = os.path.join(args.out_dir, f'session_{i+1}')
        else:
            sub_out_dir = args.out_dir
        if not os.path.isdir(sub_out_dir):
            os.mkdir(sub_out_dir)

        # Get the bundle
        new_sft = StatefulTractogram.from_sft(sft.streamlines, sft_list[0])

        # Clean its streamlines
        timer = time.time()
        cut_sft = cut_streamlines_with_mask(
            new_sft, binary_mask,
            cutting_style=CuttingStyle.KEEP_LONGEST)
        cut_sft = remove_overlapping_points_streamlines(cut_sft,
                                                        args.threshold)
        cut_sft = filter_streamlines_by_nb_points(cut_sft, min_nb_points=2)
        logging.debug(
            f'Cut streamlines in {round(time.time() - timer, 3)} seconds')

        # Prepare the 'color' array. Will not contain the right data yet, we
        # will change it later based on what we want to save. But will set the
        # right shape.
        cut_sft.data_per_point['color'] = ArraySequence(cut_sft.streamlines)

        # Save labels, distance and correlation files and save tractograms with
        # these values as color.
        ### toDo. Why does it use binary_list as mask? The correlation_thr is
        #    not applied on that mask, but streamlines_thr yes!
        bundle_mask = binary_list[i]

        # -- labels
        save_output(sub_out_dir, 'labels', labels_map.astype(np.uint16),
                    cut_sft, ref_affine, bundle_mask, cmap,
                    max_val=args.nb_pts)

        # -- distance
        save_output(sub_out_dir, 'distance', distance_map.astype(np.float32),
                    cut_sft, ref_affine, bundle_mask, cmap)

        # -- correlation
        # Only saving correlation tractogram if more than one bundle
        save_output(sub_out_dir, 'correlation', corr_map.astype(np.float32),
                    cut_sft, ref_affine, bundle_mask, cmap, max_val=1,
                    save_trk=True if len(args.in_bundles) > 1 else False)


if __name__ == '__main__':
    main()
