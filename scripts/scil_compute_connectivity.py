#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script computes a variety of measures in the form of connectivity
matrices. This script is made to follow scil_decompose_connectivity and
uses the same labels list as input.

The script expects a folder containing all relevants bundles following the
naming convention LABEL1_LABEL2.trk and a text file containing the list of
labels that should be part of the matrices. The ordering of labels in the
matrices will follow the same order as the list.
This script only generates matrices in the form of array, does not visualize
or reorder the labels (node).

The parameter --similarity expects a folder with density maps (LABEL1_LABEL2.nii.gz)
following the same naming convention as the input directory.
The bundles should be averaged version in the same space. This will
compute the weighted-dice between each node and their homologuous average
version.

The parameters --metrics can be used more than once and expect a map (t1, fa,
etc.) in the same space and each will generate a matrix. The average value in
the volume occupied by the bundle will be reported in the matrices nodes.

The parameters --maps can be used more than once and expect a folder with
pre-computed maps (LABEL1_LABEL2.nii.gz) following the same naming convention
as the input directory. Each will generate a matrix. The average non-zeros
value in the map will be reported in the matrices nodes.
"""

import argparse
import copy
import itertools
import multiprocessing
import logging
import os

import coloredlogs
from dipy.io.utils import is_header_compatible, get_reference_info
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamlinespeed import length
import nibabel as nib
import numpy as np

from scilpy.tractanalysis.reproducibility_measures import compute_bundle_adjacency_voxel
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_verbose_arg, add_reference_arg,
                             assert_inputs_exist, assert_outputs_exist)


def load_node_nifti(directory, in_label, out_label, ref_filename):
    in_filename_1 = os.path.join(directory,
                                 '{}_{}.nii.gz'.format(in_label, out_label))
    in_filename_2 = os.path.join(directory,
                                 '{}_{}.nii.gz'.format(out_label, in_label))
    in_filename = None
    if os.path.isfile(in_filename_1):
        in_filename = in_filename_1
    elif os.path.isfile(in_filename_2):
        in_filename = in_filename_2

    if in_filename is not None:
        if not is_header_compatible(in_filename, ref_filename):
            logging.error('{} and {} do not have a compatible header'.format(
                in_filename, ref_filename))
            raise IOError
        return nib.load(in_filename).get_fdata()

    _, dims, _, _ = get_reference_info(ref_filename)
    return np.zeros(dims)


def _processing_wrapper(args):
    bundles_dir = args[0]
    in_label, out_label = args[1]
    measures_to_compute = copy.copy(args[2])
    weighted = args[3]
    if args[4] is not None:
        similarity_directory = args[4][0]

    in_filename_1 = os.path.join(bundles_dir,
                                 '{}_{}.trk'.format(in_label, out_label))
    in_filename_2 = os.path.join(bundles_dir,
                                 '{}_{}.trk'.format(out_label, in_label))
    if os.path.isfile(in_filename_1):
        in_filename = in_filename_1
    elif os.path.isfile(in_filename_2):
        in_filename = in_filename_2
    else:
        return

    sft = load_tractogram(in_filename, 'same')
    affine, dimensions, voxel_sizes, _ = sft.space_attributes
    measures_to_return = {}

    # Precompute to save one transformation, insert later
    if 'length' in measures_to_compute:
        streamlines_copy = list(sft.get_streamlines_copy())
        mean_length = np.average(length(streamlines_copy))

    # If density is not required, do not compute it
    # Only required for volume, similarity and any metrics
    if not ((len(measures_to_compute) == 1 and
             ('length' in measures_to_compute or
              'streamline_count' in measures_to_compute)) or
            (len(measures_to_compute) == 2 and
             ('length' in measures_to_compute and
              'streamline_count' in measures_to_compute))):
        sft.to_vox()
        sft.to_corner()
        density = compute_tract_counts_map(sft.streamlines,
                                           dimensions)

    if 'volume' in measures_to_compute:
        measures_to_return['volume'] = np.count_nonzero(density) * \
            np.prod(voxel_sizes)
        measures_to_compute.remove('volume')
    if 'streamline_count' in measures_to_compute:
        measures_to_return['streamline_count'] = len(sft)
        measures_to_compute.remove('streamline_count')
    if 'length' in measures_to_compute:
        measures_to_return['length'] = mean_length
        measures_to_compute.remove('length')
    if 'similarity' in measures_to_compute and similarity_directory:
        density_sim = load_node_nifti(similarity_directory,
                                      in_label, out_label,
                                      in_filename)

        ba_vox = compute_bundle_adjacency_voxel(density, density_sim)

        measures_to_return['similarity'] = ba_vox
        measures_to_compute.remove('similarity')

    for measure in measures_to_compute:
        if os.path.isdir(measure):
            map_dirname = measure
            map_data = load_node_nifti(
                map_dirname, in_label, out_label, in_filename)
            measures_to_return[map_dirname] = np.average(
                map_data[map_data > 0])
        elif os.path.isfile(measure):
            metric_filename = measure
            if not is_header_compatible(metric_filename, sft):
                raise IOError('{} and {} do not have a compatible header'.format(
                    in_filename, metric_filename))

            metric_data = nib.load(metric_filename).get_fdata()
            if weighted:
                density = density / np.max(density)
                voxels_value = metric_data * density
                voxels_value = voxels_value[voxels_value > 0]
            else:
                voxels_value = metric_data[density > 0]

            measures_to_return[metric_filename] = np.average(voxels_value)

    return {(in_label, out_label): measures_to_return}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,)
    p.add_argument('in_bundles_dir',
                   help='Folder containing all the bundle files (.trk).')
    p.add_argument('labels_list',
                   help='Text file containing the list of labels from the '
                        'atlas.')
    p.add_argument('--volume', metavar='OUT_FILE',
                   help='Output file for the volume weighted matrix (.npy).')
    p.add_argument('--streamline_count', metavar='OUT_FILE',
                   help='Output file for the streamline count weighted matrix '
                        '(.npy).')
    p.add_argument('--length', metavar='OUT_FILE',
                   help='Output file for the length weighted matrix (.npy).')
    p.add_argument('--similarity', nargs=2,
                   metavar=('IN_FOLDER', 'OUT_FILE'),
                   help='Input folder containing the averaged bundle density\n'
                        'maps (.nii.gz) and output file for the similarity '
                        'weighted matrix (.npy).')
    p.add_argument('--maps', nargs=2,  action='append',
                   metavar=('IN_FOLDER', 'OUT_FILE'),
                   help='Input folder containing pre-computed maps (.nii.gz)\n'
                        'and output file for the weighted matrix (.npy).')
    p.add_argument('--metrics', nargs=2, action='append',
                   metavar=('IN_FILE', 'OUT_FILE'),
                   help='Input (.nii.gz). and output file (.npy) for a metric '
                        'weighted matrix.')

    p.add_argument('--density_weighting', action="store_true",
                   help='Use density-weighting for the metric weighted matrix.')
    p.add_argument('--no_self_connection', action="store_true",
                   help='Eliminate the diagonal from the matrices.')

    add_processes_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.labels_list)
    if not os.path.isdir(args.in_bundles_dir):
        parser.error('The directory {} does not exist.'.format(
            args.in_bundles_dir))

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)
    coloredlogs.install(level=log_level)

    measures_to_compute = []
    measures_output_filename = []
    if args.volume:
        measures_to_compute.append('volume')
        measures_output_filename.append(args.volume)
    if args.streamline_count:
        measures_to_compute.append('streamline_count')
        measures_output_filename.append(args.streamline_count)
    if args.length:
        measures_to_compute.append('length')
        measures_output_filename.append(args.length)
    if args.similarity:
        measures_to_compute.append('similarity')
        measures_output_filename.append(args.similarity[1])

    dict_maps_out_name = {}
    if args.maps is not None:
        for in_folder, out_name in args.maps:
            measures_to_compute.append(in_folder)
            dict_maps_out_name[in_folder] = out_name
            measures_output_filename.append(out_name)

    dict_metrics_out_name = {}
    if args.metrics is not None:
        for in_name, out_name in args.metrics:
            # Verify that all metrics are compatible with each other
            if not is_header_compatible(args.metrics[0][0], in_name):
                raise IOError('Metrics do not share a compatible header'.format(
                    args.metrics[0][0], in_name))

            # This is necessary to support more than one map for weighting
            measures_to_compute.append(in_name)
            dict_metrics_out_name[in_name] = out_name
            measures_output_filename.append(out_name)

    assert_outputs_exist(parser, args, measures_output_filename)
    if not measures_to_compute:
        parser.error('No connectivity measures were selected, nothing'
                     'to compute.')
    logging.info('The following measures will be computed and save: {}'.format(
        measures_to_compute))

    labels_list = np.loadtxt(args.labels_list, dtype=int).tolist()

    comb_list = list(itertools.combinations(labels_list, r=2))
    if not args.no_self_connection:
        comb_list.extend(zip(labels_list, labels_list))

    pool = multiprocessing.Pool(args.nbr_processes)
    measures_dict_list = pool.map(_processing_wrapper,
                                  zip(itertools.repeat(args.in_bundles_dir),
                                      comb_list,
                                      itertools.repeat(measures_to_compute),
                                      itertools.repeat(args.density_weighting),
                                      itertools.repeat(args.similarity)))

    # Removing None entries (combinaisons that do not exist)
    # Fusing the multiprocessing output into a single dictionary
    measures_dict_list = [it for it in measures_dict_list if it is not None]
    measures_dict = measures_dict_list[0]
    for dix in measures_dict_list[1:]:
        measures_dict.update(dix)

    if args.no_self_connection:
        total_elem = len(measures_dict.keys())*2
    else:
        total_elem = len(measures_dict.keys())*2 - len(labels_list)

    logging.info('Out of {} node, {} contain values'.format(
        total_elem, len(measures_dict.keys())*2))

    # Filling out all the matrices (symmetric) in the order of labels_list
    nbr_of_measures = len(measures_to_compute)
    matrix = np.zeros((len(labels_list), len(labels_list), nbr_of_measures))
    for in_label, out_label in measures_dict:
        curr_node_dict = measures_dict[(in_label, out_label)]
        measures_ordering = list(curr_node_dict.keys())
        for i, measure in enumerate(curr_node_dict):
            in_pos = labels_list.index(in_label)
            out_pos = labels_list.index(out_label)
            matrix[in_pos, out_pos, i] = curr_node_dict[measure]
            matrix[out_pos, in_pos, i] = curr_node_dict[measure]

    # Saving the matrices separatly with the specified name
    for i, measure in enumerate(measures_ordering):
        if measure == 'volume':
            matrix_basename = args.volume
        elif measure == 'streamline_count':
            matrix_basename = args.streamline_count
        elif measure == 'length':
            matrix_basename = args.length
        elif measure == 'similarity':
            matrix_basename = args.similarity[1]
        elif measure in dict_metrics_out_name:
            matrix_basename = dict_metrics_out_name[measure]
        elif measure in dict_maps_out_name:
            matrix_basename = dict_maps_out_name[measure]

        np.save(matrix_basename, matrix[:, :, i])


if __name__ == "__main__":
    main()
