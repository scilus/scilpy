#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate pair-wise similarity measures of masks and atlas.
All volumes must be co-registered in the same space.

Support multiple input volume. The following command will compare all
combinations of the input volumes (1-2, 1-3, 2-3):
  scil_volume_pairwise_comparison.py mask1.nii.gz mask2.nii.gz \
    mask3.nii.gz out.json

The following command will compare all input of the input volumes to a single
volume (1-ref, 2-ref, 3-ref):
  scil_volume_pairwise_comparison.py mask1.nii.gz mask2.nii.gz \
    mask3.nii.gz out.json --single_compare ref.nii.gz

This can work for BET mask, WMPARC, bundle label maps. The datatype of the
input volumes must be uint8 (mask) or uint16 (label map and atlas).
The computed similarity measures are:
    adjacency_voxels, dice_voxels, volume_overlap, volume_overreach.
For each measure, an entry in the json file will be created and for each unique
value present in the input volumes there will be an entry under the measure.
(i.e. a binary mask will have one entries for each measure, 1).

If you have streamlines to compare, the following script could be
of interest for you: scil_bundle_pairwise_comparison.py
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
                             add_verbose_arg,
                             assert_headers_compatible,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             validate_nbr_processes)
from scilpy.image.labels import get_data_as_labels
from scilpy.tractanalysis.reproducibility_measures import \
    compare_volume_wrapper
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_volumes', nargs='*',
                   help='Path of the input volumes.')
    p.add_argument('out_json',
                   help='Path of the output json file.')

    p.add_argument('--ignore_zeros_in_BA', action='store_true',
                   help='If set, do not count zeros in the average bundle '
                        'adjacency (BA).')

    p.add_argument('--single_compare', metavar='FILE',
                   help='Compare inputs to this single file.')
    p.add_argument('--ratio', action='store_true',
                   help='Compute overlap and overreach as a ratio over the '
                        'reference volume rather than volume.\n'
                        'Can only be used if also using --single_compare`.')
    p.add_argument('--labels_to_mask', action='store_true',
                   help='Allows for comparison between labels and single '
                        'binary mask. Can only be used with '
                        '--single_compare.')

    add_processes_arg(p)
    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def compute_all_measures(args):
    filename_1 = args[0][0]
    filename_2 = args[0][1]
    adjency_no_overlap = args[1]
    ratio = args[2]
    labels_to_mask = args[3]

    img_1, dtype_1 = load_img(filename_1)

    if np.issubdtype(dtype_1, np.floating):
        raise ValueError('Input file {} is of type float.\n'
                         'Please convert to uint8 (mask) or '
                         'uint16 (labels).'.format(filename_1))
    img_2, dtype_2 = load_img(filename_2)
    if np.issubdtype(dtype_2, np.floating):
        raise ValueError('Input file {} is of type float.\n'
                         'Please convert to uint8 (mask) or '
                         'uint16 (labels).'.format(filename_2))

    data_1 = get_data_as_labels(img_1)
    data_2 = get_data_as_labels(img_2)

    _, _, voxel_size, _ = get_reference_info(img_1)
    voxel_size = np.prod(voxel_size)
    logging.info(f"Comparing {filename_1} and {filename_2}")
    dict_measures = compare_volume_wrapper(data_1, data_2, voxel_size,
                                           ratio, adjency_no_overlap,
                                           labels_to_mask)
    return dict_measures


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_volumes)
    assert_outputs_exist(parser, args, [args.out_json])
    assert_headers_compatible(parser, args.in_volumes)

    if args.ratio and not args.single_compare:
        parser.error('Can only compute ratio if also using `single_compare`')

    if args.labels_to_mask and not args.single_compare:
        parser.error('Can only compare labels to a mask if also using '
                     '`single_compare`.')

    nbr_cpu = validate_nbr_processes(parser, args)
    if nbr_cpu > 1:
        pool = multiprocessing.Pool(nbr_cpu)

    if args.single_compare:
        # Remove the single_compare from inputs and combine it to all
        # other files.
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
                args.ignore_zeros_in_BA,
                args.ratio,
                args.labels_to_mask]))
    else:
        all_measures_dict = pool.map(
            compute_all_measures,
            zip(comb_dict_keys,
                itertools.repeat(args.ignore_zeros_in_BA),
                itertools.repeat(args.ratio),
                itertools.repeat(args.labels_to_mask)))
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
