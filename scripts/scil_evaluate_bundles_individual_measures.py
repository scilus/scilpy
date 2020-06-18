#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate basic measurements of bundles, all at once.
All tractograms must be in the same space (aligned to one reference)

The computed measures are:
volume, volume_endpoints, streamlines_count, avg_length, std_length,
min_length, max_length, mean_curvature
"""

import argparse
import json
import logging
import multiprocessing

from dipy.io.streamline import load_tractogram
from dipy.tracking.metrics import mean_curvature
from dipy.tracking.utils import length
import numpy as np

from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_processes_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             link_bundles_and_reference,
                             validate_nbr_processes)

from scilpy.tractanalysis.reproducibility_measures import get_endpoints_density_map
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundles', nargs='+',
                   help='Path of the input bundles.')
    p.add_argument('out_json',
                   help='Path of the output file.')

    add_reference_arg(p)
    add_processes_arg(p)
    add_json_args(p)
    add_overwrite_arg(p)

    return p


def compute_measures(filename_tuple):
    sft = load_tractogram(filename_tuple[0], filename_tuple[1])
    _, dimensions, voxel_size, _ = sft.space_attributes

    nbr_streamlines = len(sft)
    if not nbr_streamlines:
        logging.warning('{} is empty'.format(filename_tuple[0]))
        return dict(zip(['volume', 'volume_endpoints', 'streamlines_count',
                         'avg_length', 'std_length', 'min_length', 'max_length',
                         'mean_curvature'], [0, 0, 0, 0, 0, 0, 0, 0]))

    length_list = list(length(list(sft.streamlines)))
    length_avg = float(np.average(length_list))
    length_std = float(np.std(length_list))
    length_min = float(np.min(length_list))
    length_max = float(np.max(length_list))

    sft.to_vox()
    sft.to_corner()
    streamlines = sft.streamlines
    density = compute_tract_counts_map(streamlines, dimensions)
    endpoints_density = get_endpoints_density_map(streamlines, dimensions)

    curvature_list = np.zeros((nbr_streamlines,))
    for i in range(nbr_streamlines):
        curvature_list[i] = mean_curvature(sft.streamlines[i])

    return dict(zip(['volume', 'volume_endpoints', 'streamlines_count',
                     'avg_length', 'std_length', 'min_length', 'max_length',
                     'mean_curvature'],
                    [np.count_nonzero(density)*np.product(voxel_size),
                     np.count_nonzero(endpoints_density) *
                     np.product(voxel_size),
                     nbr_streamlines, length_avg, length_std, length_min,
                     length_max, float(np.mean(curvature_list))]))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, args.in_bundles)
    assert_outputs_exist(parser, args, args.out_json)

    nbr_cpu = validate_nbr_processes(parser, args)
    bundles_references_tuple_extended = link_bundles_and_reference(
        parser, args, args.in_bundles)

    if nbr_cpu == 1:
        all_measures_dict = []
        for i in bundles_references_tuple_extended:
            all_measures_dict.append(compute_measures(i))
    else:
        pool = multiprocessing.Pool(nbr_cpu)
        all_measures_dict = pool.map(compute_measures,
                                     bundles_references_tuple_extended)
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
                    measure_dict[measure_name])

    with open(args.out_json, 'w') as outfile:
        json.dump(output_measures_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
