#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate pair-wise similarity measures of bundles.
All tractograms must be in the same space (aligned to one reference).

For the voxel representation, the computed similarity measures are:
    bundle_adjacency_voxels, dice_voxels, w_dice_voxels, density_correlation
    volume_overlap, volume_overreach
The same measures are also evluated for the endpoints.

For the streamline representation, the computed similarity measures are:
    bundle_adjacency_streamlines, dice_streamlines, streamlines_count_overlap,
    streamlines_count_overreach

Formerly: scil_evaluate_bundles_pairwise_agreement_measures.py
"""

import argparse
import copy
import hashlib
import itertools
import json
import logging
import multiprocessing
import os
import shutil

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.utils import is_header_compatible, get_reference_info
from dipy.segment.clustering import qbx_and_merge
import nibabel as nib
import numpy as np
from numpy.random import RandomState

from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_processes_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             link_bundles_and_reference,
                             validate_nbr_processes, assert_headers_compatible)
from scilpy.tractanalysis.reproducibility_measures \
    import (compute_dice_voxel,
            compute_bundle_adjacency_streamlines,
            compute_bundle_adjacency_voxel,
            compute_correlation,
            compute_dice_streamlines)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map

from scilpy.tractograms.streamline_and_mask_operations import \
    get_endpoints_density_map


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundles', nargs='+',
                   help='Path of the input bundles.')
    p.add_argument('out_json',
                   help='Path of the output json file.')
    p.add_argument('--streamline_dice', action='store_true',
                   help='Compute streamline-wise dice coefficient.\n'
                        'Tractograms must be identical [%(default)s].')
    p.add_argument('--bundle_adjency_no_overlap', action='store_true',
                   help='If set, do not count zeros in the average BA.')
    p.add_argument('--disable_streamline_distance', action='store_true',
                   help='Will not compute the streamlines distance \n'
                        '[%(default)s].')
    p.add_argument('--single_compare',
                   help='Compare inputs to this single file.')
    p.add_argument('--keep_tmp', action='store_true',
                   help='Will not delete the tmp folder at the end.')
    p.add_argument('--ratio', action='store_true',
                   help='Compute overlap and overreach as a ratio over the\n'
                        'reference tractogram in a Tractometer-style way.\n'
                        'Can only be used if also using the `single_compare` '
                        'option.')

    add_processes_arg(p)
    add_reference_arg(p)
    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def load_data_tmp_saving(args):
    filename = args[0]
    reference = args[1]
    init_only = args[2]
    disable_centroids = args[3]

    # Since data is often re-used when comparing multiple bundles, anything
    # that can be computed once is saved temporarily and simply loaded on
    # demand
    hash_tmp = hashlib.md5(filename.encode()).hexdigest()
    tmp_density_filename = os.path.join('tmp_measures/',
                                        '{}_density.nii.gz'.format(hash_tmp))
    tmp_endpoints_filename = os.path.join('tmp_measures/',
                                          '{}_endpoints.nii.gz'.format(
                                                                    hash_tmp))
    tmp_centroids_filename = os.path.join('tmp_measures/',
                                          '{}_centroids.trk'.format(hash_tmp))

    sft = load_tractogram(filename, reference)
    sft.to_vox()
    sft.to_corner()
    streamlines = sft.get_streamlines_copy()
    if not streamlines:
        if init_only:
            logging.warning('{} is empty'.format(filename))
        return None

    if os.path.isfile(tmp_density_filename) \
            and os.path.isfile(tmp_endpoints_filename) \
            and os.path.isfile(tmp_centroids_filename):
        # If initilization, loading the data is useless
        if init_only:
            return None
        density = nib.load(tmp_density_filename).get_fdata(dtype=np.float32)
        endpoints_density = nib.load(
            tmp_endpoints_filename).get_fdata(dtype=np.float32)
        sft_centroids = load_tractogram(tmp_centroids_filename, reference)
        sft_centroids.to_vox()
        sft_centroids.to_corner()
        centroids = sft_centroids.get_streamlines_copy()
    else:
        transformation, dimensions, _, _ = sft.space_attributes
        density = compute_tract_counts_map(streamlines, dimensions)
        endpoints_density = get_endpoints_density_map(sft, point_to_select=3)
        thresholds = [32, 24, 12, 6]
        if disable_centroids:
            centroids = []
        else:
            centroids = qbx_and_merge(streamlines, thresholds,
                                      rng=RandomState(0),
                                      verbose=False).centroids

        # Saving tmp files to save on future computation
        nib.save(nib.Nifti1Image(density.astype(np.float32), transformation),
                 tmp_density_filename)
        nib.save(nib.Nifti1Image(endpoints_density.astype(np.int16),
                                 transformation),
                 tmp_endpoints_filename)

        # Saving in vox space and corner.
        centroids_sft = StatefulTractogram.from_sft(centroids, sft)
        save_tractogram(centroids_sft, tmp_centroids_filename)

    return density, endpoints_density, streamlines, centroids


def compute_all_measures(args):
    tuple_1, tuple_2 = args[0]
    filename_1, reference_1 = tuple_1
    filename_2, reference_2 = tuple_2
    streamline_dice = args[1]
    bundle_adjency_no_overlap = args[2]
    disable_streamline_distance = args[3]
    ratio = args[4]

    data_tuple_1 = load_data_tmp_saving([filename_1, reference_1, False,
                                         disable_streamline_distance])
    if data_tuple_1 is None:
        return None

    density_1, endpoints_density_1, bundle_1, \
        centroids_1 = data_tuple_1

    data_tuple_2 = load_data_tmp_saving([filename_2, reference_2, False,
                                         disable_streamline_distance])
    if data_tuple_2 is None:
        return None

    density_2, endpoints_density_2, bundle_2, \
        centroids_2 = data_tuple_2

    _, _, voxel_size, _ = get_reference_info(reference_1)
    voxel_size = np.product(voxel_size)

    # These measures are in mm^3
    binary_1 = copy.copy(density_1)
    binary_1[binary_1 > 0] = 1
    binary_2 = copy.copy(density_2)
    binary_2[binary_2 > 0] = 1
    volume_overlap = np.count_nonzero(binary_1 * binary_2)
    volume_overlap_endpoints = np.count_nonzero(
        endpoints_density_1 * endpoints_density_2)
    volume_overreach = np.abs(np.count_nonzero(
        binary_1 + binary_2) - volume_overlap)
    volume_overreach_endpoints = np.abs(np.count_nonzero(
        endpoints_density_1 + endpoints_density_2) - volume_overlap_endpoints)

    if ratio:
        count = np.count_nonzero(binary_1)
        volume_overlap /= count
        volume_overlap_endpoints /= count
        volume_overreach /= count
        volume_overreach_endpoints /= count

    # These measures are in mm
    bundle_adjacency_voxel = compute_bundle_adjacency_voxel(
        density_1, density_2,
        non_overlap=bundle_adjency_no_overlap)
    if streamline_dice and not disable_streamline_distance:
        bundle_adjacency_streamlines = \
            compute_bundle_adjacency_streamlines(
                bundle_1, bundle_2,
                non_overlap=bundle_adjency_no_overlap)
    elif not disable_streamline_distance:
        bundle_adjacency_streamlines = \
            compute_bundle_adjacency_streamlines(
                bundle_1, bundle_2,
                centroids_1=centroids_1,
                centroids_2=centroids_2,
                non_overlap=bundle_adjency_no_overlap)
    # These measures are between 0 and 1
    dice_vox, w_dice_vox = compute_dice_voxel(density_1, density_2)

    dice_vox_endpoints, w_dice_vox_endpoints = compute_dice_voxel(
        endpoints_density_1,
        endpoints_density_2)
    density_correlation = compute_correlation(density_1, density_2)
    density_correlation_endpoints = compute_correlation(endpoints_density_1,
                                                        endpoints_density_2)

    measures_name = ['bundle_adjacency_voxels',
                     'dice_voxels', 'w_dice_voxels',
                     'volume_overlap',
                     'volume_overreach',
                     'dice_voxels_endpoints',
                     'w_dice_voxels_endpoints',
                     'volume_overlap_endpoints',
                     'volume_overreach_endpoints',
                     'density_correlation',
                     'density_correlation_endpoints']

    # If computing ratio, voxel size does not make sense
    if ratio:
        voxel_size = 1.
    measures = [bundle_adjacency_voxel,
                dice_vox, w_dice_vox,
                volume_overlap * voxel_size,
                volume_overreach * voxel_size,
                dice_vox_endpoints,
                w_dice_vox_endpoints,
                volume_overlap_endpoints * voxel_size,
                volume_overreach_endpoints * voxel_size,
                density_correlation,
                density_correlation_endpoints]

    if not disable_streamline_distance:
        measures_name += ['bundle_adjacency_streamlines']
        measures += [bundle_adjacency_streamlines]

    # Only when the tractograms are exactly the same
    if streamline_dice:
        dice_streamlines, streamlines_intersect, streamlines_union = \
            compute_dice_streamlines(bundle_1, bundle_2)
        streamlines_count_overlap = len(streamlines_intersect)
        streamlines_count_overreach = len(
            streamlines_union) - len(streamlines_intersect)
        measures_name += ['dice_streamlines',
                          'streamlines_count_overlap',
                          'streamlines_count_overreach']
        measures += [dice_streamlines,
                     streamlines_count_overlap,
                     streamlines_count_overreach]

    return dict(zip(measures_name, measures))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bundles, args.reference)
    assert_outputs_exist(parser, args, [args.out_json])
    assert_headers_compatible(parser, args.in_bundles,
                              reference=args.reference)

    if args.ratio and not args.single_compare:
        parser.error('Can only compute ratio if also using `single_compare`')

    nbr_cpu = validate_nbr_processes(parser, args)
    if nbr_cpu > 1:
        pool = multiprocessing.Pool(nbr_cpu)

    if not os.path.isdir('tmp_measures/'):
        os.mkdir('tmp_measures/')

    if args.single_compare:
        # Move the single_compare only once, at the end.
        if args.single_compare in args.in_bundles:
            args.in_bundles.remove(args.single_compare)
        bundles_list = args.in_bundles + [args.single_compare]
        bundles_references_tuple_extended = link_bundles_and_reference(
            parser, args, bundles_list)

        single_compare_reference_tuple = \
            bundles_references_tuple_extended.pop()
        comb_dict_keys = list(itertools.product(
                                        bundles_references_tuple_extended,
                                        [single_compare_reference_tuple]))
    else:
        bundles_list = args.in_bundles
        # Pre-compute the needed files, to avoid conflict when the number
        # of cpu is higher than the number of bundle
        bundles_references_tuple = link_bundles_and_reference(parser,
                                                              args,
                                                              bundles_list)

        # This approach is only so pytest can run
        if nbr_cpu == 1:
            for i in range(len(bundles_references_tuple)):
                load_data_tmp_saving([bundles_references_tuple[i][0],
                                      bundles_references_tuple[i][1],
                                      True, args.disable_streamline_distance])
        else:
            pool.map(load_data_tmp_saving,
                     zip([tup[0] for tup in bundles_references_tuple],
                         [tup[1] for tup in bundles_references_tuple],
                         itertools.repeat(True),
                         itertools.repeat(args.disable_streamline_distance)))

        comb_dict_keys = list(itertools.combinations(
            bundles_references_tuple, r=2))

    if nbr_cpu == 1:
        all_measures_dict = []
        for i in comb_dict_keys:
            all_measures_dict.append(compute_all_measures([
                i, args.streamline_dice,
                args.bundle_adjency_no_overlap,
                args.disable_streamline_distance,
                args.ratio]))
    else:
        all_measures_dict = pool.map(
            compute_all_measures,
            zip(comb_dict_keys,
                itertools.repeat(args.streamline_dice),
                itertools.repeat(args.bundle_adjency_no_overlap),
                itertools.repeat(args.disable_streamline_distance),
                itertools.repeat(args.ratio)))
        pool.close()
        pool.join()

    output_measures_dict = {}
    for measure_dict in all_measures_dict:
        # Empty bundle should not make the script crash
        if measure_dict is not None:
            for measure_name in measure_dict.keys():
                # Create an empty list first
                if measure_name not in output_measures_dict:
                    output_measures_dict[measure_name] = []
                output_measures_dict[measure_name].append(
                    float(measure_dict[measure_name]))

    with open(args.out_json, 'w') as outfile:
        json.dump(output_measures_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)

    if not args.keep_tmp:
        shutil.rmtree('tmp_measures/')


if __name__ == "__main__":
    main()
