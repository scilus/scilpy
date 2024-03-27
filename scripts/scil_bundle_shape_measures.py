#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate basic measurements of bundle(s).

The computed measures are:
    - volume_info: volume, volume_endpoints
    - streamlines_info: streamlines_count, avg_length (in mm or in number of
      point), average step size, min_length, max_length.
      ** You may also get this information with scil_tractogram_print_info.py.
    - shape_info: span, curl, diameter, elongation, surface area,
      irregularity, end surface area, radius, end surface irregularity,
      mean_curvature, fractal dimension.
      ** The diameter, here, is a simple estimation using volume / length.
      For a more complex calculation, see scil_bundle_diameter.py.

With more than one bundle, the measures are averaged over bundles. All
tractograms must be in the same space.

The set average contains the average measures of all input bundles. The
measures that are dependent on the streamline count are weighted by the number
of streamlines of each bundle. Each of these average measure is computed by
first summing the multiple of a measure and the streamline count of each
bundle and divide the sum by the total number of streamlines. Thus, measures
including length and span are essentially averages of all the streamlines.
Other streamline-related set measure are computed with other set averages.
Whereas bundle-related measures are computed as an average of all bundles.
These measures include volume and surface area.

The fractal dimension is dependent on the voxel size and the number of voxels.
If data comparison is performed, the bundles MUST be in same resolution.

Formerly: scil_compute_bundle_volume.py or
scil_evaluate_bundles_individual_measures.py
"""

import argparse
import itertools
import json
import logging
import multiprocessing

from dipy.io.streamline import load_tractogram
from dipy.tracking.metrics import mean_curvature
from dipy.tracking.utils import length
import numpy as np

from scilpy.io.utils import (add_json_args, add_verbose_arg,
                             add_overwrite_arg, add_processes_arg,
                             add_reference_arg, assert_inputs_exist,
                             assert_outputs_exist, link_bundles_and_reference,
                             validate_nbr_processes, assert_headers_compatible)
from scilpy.tractanalysis.bundle_operations import uniformize_bundle_sft
from scilpy.tractanalysis.reproducibility_measures \
    import (approximate_surface_node,
            compute_fractal_dimension)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractograms.streamline_and_mask_operations import \
    get_endpoints_density_map, get_head_tail_density_maps

EPILOG = """
References:
[1] Fang-Cheng Yeh. 2020.
    Shape analysis of the human association pathways. NeuroImage.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundles', nargs='+',
                   help='Path of the input bundles.')
    p.add_argument('--out_json',
                   help='Path of the output file. If not given, the output '
                        'is simply printed on screen.')
    p.add_argument('--group_statistics', action='store_true',
                   help='Show average measures [%(default)s].')

    p.add_argument('--no_uniformize', action='store_true',
                   help='Do NOT automatically uniformize endpoints for the'
                        'endpoints related metrics.')
    add_reference_arg(p)
    add_processes_arg(p)
    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def compute_measures(args):
    filename_tuple, no_uniformize = args
    sft = load_tractogram(filename_tuple[0], filename_tuple[1])
    _, dimensions, voxel_size, _ = sft.space_attributes
    if not no_uniformize:
        uniformize_bundle_sft(sft)
    nbr_streamlines = len(sft)
    if not nbr_streamlines:
        logging.warning('{} is empty'.format(filename_tuple[0]))
        return dict(zip(['volume', 'volume_endpoints', 'streamlines_count',
                         'avg_length', 'std_length', 'min_length',
                         'max_length', 'span', 'curl', 'diameter',
                         'elongation', 'surface_area', 'end_surface_area_head',
                         'end_surface_area_tail', 'radius_head', 'radius_tail',
                         'irregularity', 'irregularity_of_end_surface_head',
                         'irregularity_of_end_surface_tail', 'mean_curvature',
                         'fractal_dimension'],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0]))

    streamline_cords = list(sft.streamlines)
    length_list = list(length(streamline_cords))
    length_avg = float(np.average(length_list))
    length_std = float(np.std(length_list))
    length_min = float(np.min(length_list))
    length_max = float(np.max(length_list))

    sft.to_vox()
    sft.to_corner()
    streamlines = sft.streamlines
    density = compute_tract_counts_map(streamlines, dimensions)
    endpoints_density = get_endpoints_density_map(sft)

    span_list = list(map(compute_span, streamline_cords))
    span = float(np.average(span_list))
    curl = length_avg / span
    volume = np.count_nonzero(density) * np.product(voxel_size)
    diameter = 2 * np.sqrt(volume / (np.pi * length_avg))
    elon = length_avg / diameter

    roi = np.where(density != 0, 1, density)
    surf_area = approximate_surface_node(roi) * (voxel_size[0] ** 2)
    irregularity = surf_area / (np.pi * diameter * length_avg)

    endpoints_map_head, endpoints_map_tail = \
        get_head_tail_density_maps(sft)
    endpoints_map_head_roi = \
        np.where(endpoints_map_head != 0, 1, endpoints_map_head)
    endpoints_map_tail_roi = \
        np.where(endpoints_map_tail != 0, 1, endpoints_map_tail)
    end_sur_area_head = \
        approximate_surface_node(endpoints_map_head_roi) * (voxel_size[0] ** 2)
    end_sur_area_tail = \
        approximate_surface_node(endpoints_map_tail_roi) * (voxel_size[0] ** 2)

    endpoints_coords_head = np.array(np.where(endpoints_map_head_roi)).T
    endpoints_coords_tail = np.array(np.where(endpoints_map_tail_roi)).T
    radius_head = 1.5 * np.average(
        np.sqrt(((endpoints_coords_head - np.average(
            endpoints_coords_head, axis=0))
            ** 2).sum(axis=1)))
    radius_tail = 1.5 * np.average(
        np.sqrt(((endpoints_coords_tail - np.average(
            endpoints_coords_tail, axis=0))
            ** 2).sum(axis=1)))
    end_irreg_head = (np.pi * radius_head ** 2) / end_sur_area_head
    end_irreg_tail = (np.pi * radius_tail ** 2) / end_sur_area_tail

    fractal_dimension = compute_fractal_dimension(density)

    curvature_list = np.zeros((nbr_streamlines,))
    for i in range(nbr_streamlines):
        curvature_list[i] = mean_curvature(sft.streamlines[i])

    return dict(zip(['volume', 'volume_endpoints', 'streamlines_count',
                     'avg_length', 'std_length', 'min_length', 'max_length',
                     'span', 'curl', 'diameter', 'elongation', 'surface_area',
                     'end_surface_area_head', 'end_surface_area_tail',
                     'radius_head', 'radius_tail',
                     'irregularity', 'irregularity_of_end_surface_head',
                     'irregularity_of_end_surface_tail', 'mean_curvature',
                     'fractal_dimension'],
                    [volume, np.count_nonzero(endpoints_density) *
                     np.product(voxel_size), nbr_streamlines,
                     length_avg, length_std, length_min, length_max,
                     span, curl, diameter, elon, surf_area, end_sur_area_head,
                     end_sur_area_tail, radius_head, radius_tail, irregularity,
                     end_irreg_head, end_irreg_tail,
                     float(np.mean(curvature_list)), fractal_dimension]))


def compute_span(streamline_coords):
    xyz = np.asarray(streamline_coords)
    if xyz.shape[0] < 2:
        return 0
    dists = np.sqrt((np.diff([xyz[0], xyz[-1]], axis=0) ** 2).sum(axis=1))
    return np.sum(dists)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bundles, args.reference)
    assert_outputs_exist(parser, args, [], args.out_json)
    assert_headers_compatible(parser, args.in_bundles,
                              reference=args.reference)

    nbr_cpu = validate_nbr_processes(parser, args)
    bundles_references_tuple_extended = link_bundles_and_reference(
        parser, args, args.in_bundles)

    if nbr_cpu == 1:
        all_measures_dict = []
        for i in bundles_references_tuple_extended:
            all_measures_dict.append(compute_measures((i, args.no_uniformize)))
    else:
        pool = multiprocessing.Pool(nbr_cpu)
        all_measures_dict = pool.map(compute_measures,
                                     zip(bundles_references_tuple_extended,
                                         itertools.repeat(args.no_uniformize)))
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
    # add group stats if user wants
    if args.group_statistics:
        # length and span are weighted by streamline count
        group_total_length = np.sum(
            np.multiply(output_measures_dict['avg_length'],
                        output_measures_dict['streamlines_count']))
        group_total_span = np.sum(
            np.multiply(output_measures_dict['span'],
                        output_measures_dict['streamlines_count']))
        group_streamlines_count = \
            np.sum(output_measures_dict['streamlines_count'])
        group_avg_length = group_total_length / group_streamlines_count
        group_avg_span = group_total_span / group_streamlines_count
        group_avg_vol = np.average(output_measures_dict['volume'])
        group_avg_diam = \
            2 * np.sqrt(group_avg_vol / (np.pi * group_avg_length))
        output_measures_dict['group_stats'] = {}
        output_measures_dict['group_stats']['total_streamlines_count'] = \
            float(group_streamlines_count)
        output_measures_dict['group_stats']['avg_streamline_length'] = \
            group_avg_length
        # max and min length of all streamlines in all input bundles
        output_measures_dict['group_stats']['max_streamline_length'] = \
            float(np.max(output_measures_dict['max_length']))
        output_measures_dict['group_stats']['min_streamline_length'] = \
            float(np.min(output_measures_dict['min_length']))
        output_measures_dict['group_stats']['avg_streamline_span'] = \
            group_avg_span
        # computed with other set averages and not weighted by streamline count
        output_measures_dict['group_stats']['avg_volume'] = group_avg_vol
        output_measures_dict['group_stats']['avg_curl'] = \
            group_avg_length / group_avg_span
        output_measures_dict['group_stats']['avg_diameter'] = group_avg_diam
        output_measures_dict['group_stats']['avg_elongation'] = \
            group_avg_length / group_avg_diam
        output_measures_dict['group_stats']['avg_surface_area'] = \
            np.average(output_measures_dict['surface_area'])
        output_measures_dict['group_stats']['avg_irreg'] = \
            np.average(output_measures_dict['irregularity'])
        output_measures_dict['group_stats']['avg_end_surface_area_head'] = \
            np.average(output_measures_dict['end_surface_area_head'])
        output_measures_dict['group_stats']['avg_end_surface_area_tail'] = \
            np.average(output_measures_dict['end_surface_area_tail'])
        output_measures_dict['group_stats']['avg_radius_head'] = \
            np.average(output_measures_dict['radius_head'])
        output_measures_dict['group_stats']['avg_radius_tail'] = \
            np.average(output_measures_dict['radius_tail'])
        output_measures_dict['group_stats']['avg_irregularity_head'] = \
            np.average(
                output_measures_dict['irregularity_of_end_surface_head'])
        output_measures_dict['group_stats']['avg_irregularity_tail'] = \
            np.average(
                output_measures_dict['irregularity_of_end_surface_tail'])
        output_measures_dict['group_stats']['avg_fractal_dimension'] = \
            np.average(output_measures_dict['fractal_dimension'])

    if args.out_json:
        with open(args.out_json, 'w') as outfile:
            json.dump(output_measures_dict, outfile,
                      indent=args.indent, sort_keys=args.sort_keys)
    else:
        print(json.dumps(output_measures_dict,
                         indent=args.indent, sort_keys=args.sort_keys))


if __name__ == "__main__":
    main()
