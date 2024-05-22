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
import itertools
import json
import logging
import multiprocessing

from dipy.io.utils import get_reference_info
import numpy as np

from scilpy.io.image import load_img
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_processes_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_headers_compatible,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             validate_nbr_processes)
from scilpy.image.labels import get_data_as_labels
from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel, \
    compute_bundle_adjacency_voxel


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_volumes', nargs='+',
                   help='Path of the input volumes.')
    p.add_argument('out_json',
                   help='Path of the output json file.')

    p.add_argument('--adjency_no_overlap', action='store_true',
                   help='If set, do not count zeros in the average BA.')

    p.add_argument('--single_compare',
                   help='Compare inputs to this single file.')
    p.add_argument('--ratio', action='store_true',
                   help='Compute overlap and overreach as a ratio over the '
                        'reference volume.\nCan only be used if also using '
                        '--single_compare`.')

    add_processes_arg(p)
    add_reference_arg(p)
    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def compute_all_measures(args):
    filename_1 = args[0][0]
    filename_2 = args[0][1]
    adjency_no_overlap = args[1]
    ratio = args[2]

    img_1, dtype_1 = load_img(filename_1)

    if np.issubdtype(dtype_1, np.floating):
        raise ValueError('Input file {} is of type float.'.format(filename_1))
    img_2, dtype_2 = load_img(filename_2)
    if np.issubdtype(dtype_2, np.floating):
        raise ValueError('Input file {} is of type float.'.format(filename_2))

    data_1 = get_data_as_labels(img_1)
    data_2 = get_data_as_labels(img_2)

    _, _, voxel_size, _ = get_reference_info(img_1)
    voxel_size = np.product(voxel_size)

    # Exclude 0 (background)
    unique_values_1 = np.unique(data_1)[1:]
    unique_values_2 = np.unique(data_2)[1:]
    union_values = np.union1d(unique_values_1, unique_values_2)

    dict_measures = {}
    for val in union_values:
        binary_1 = np.zeros(data_1.shape, dtype=np.uint8)
        binary_1[data_1 == val] = 1
        binary_2 = np.zeros(data_2.shape, dtype=np.uint8)
        binary_2[data_2 == val] = 1

        # These measures are in mm^3
        volume_overlap = np.count_nonzero(binary_1 * binary_2)
        volume_overreach = np.abs(np.count_nonzero(
            binary_1 + binary_2) - volume_overlap)

        if ratio:
            count = np.count_nonzero(binary_1)
            volume_overlap /= count
            volume_overreach /= count

        # These measures are in mm
        bundle_adjacency_voxel = compute_bundle_adjacency_voxel(
            binary_1, binary_2,
            non_overlap=adjency_no_overlap)

        # These measures are between 0 and 1
        dice_vox, _ = compute_dice_voxel(binary_1,
                                         binary_2)

        measures_name = ['bundle_adjacency_voxels',
                         'dice_voxels',
                         'volume_overlap',
                         'volume_overreach']

        # If computing ratio, voxel size does not make sense
        if ratio:
            voxel_size = 1.
        measures = [bundle_adjacency_voxel,
                    dice_vox,
                    volume_overlap * voxel_size,
                    volume_overreach * voxel_size]

        curr_dict = dict(zip(measures_name, measures))
        for measure_name, measure in curr_dict.items():
            if measure_name not in dict_measures:
                dict_measures[measure_name] = {}
            dict_measures[measure_name].update({int(val): float(measure)})

    return dict_measures


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_volumes, args.reference)
    assert_outputs_exist(parser, args, [args.out_json])
    assert_headers_compatible(parser, args.in_volumes,
                              reference=args.reference)

    if args.ratio and not args.single_compare:
        parser.error('Can only compute ratio if also using `single_compare`')

    nbr_cpu = validate_nbr_processes(parser, args)
    if nbr_cpu > 1:
        pool = multiprocessing.Pool(nbr_cpu)

    if args.single_compare:
        # Move the single_compare only once, at the end.
        if args.single_compare in args.in_volumes:
            args.in_volumes.remove(args.single_compare)

        comb_dict_keys = list(itertools.product(args.in_volumes,
                                                [args.single_compare]))
    else:
        comb_dict_keys = list(itertools.combinations(args.in_volumes, r=2))

    if nbr_cpu == 1:
        all_measures_dict = []
        for curr_tuple in comb_dict_keys:
            all_measures_dict.append(compute_all_measures([
                curr_tuple,
                args.adjency_no_overlap,
                args.ratio]))
    else:
        all_measures_dict = pool.map(
            compute_all_measures,
            zip(comb_dict_keys,
                itertools.repeat(args.adjency_no_overlap),
                itertools.repeat(args.ratio)))
        pool.close()
        pool.join()

    output_measures_dict = {}
    for measure_dict in all_measures_dict:
        # Empty bundle should not make the script crash
        for measure_name in measure_dict.keys():
            for val in measure_dict[measure_name].keys():
                if measure_name not in output_measures_dict:
                    output_measures_dict[measure_name] = {val: []}
                elif val not in output_measures_dict[measure_name]:
                    output_measures_dict[measure_name][val] = []
                output_measures_dict[measure_name][val].append(
                    measure_dict[measure_name][val])

    with open(args.out_json, 'w') as outfile:
        json.dump(output_measures_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
