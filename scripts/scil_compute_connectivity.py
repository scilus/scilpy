#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script computes a variety of mesures in the form of connectivity
matrices. This script is made to follow scil_decompose_connectivity and the
use the same labels list as input.

Expect a folder containing all relevants bundles following the naming
convention label1_label2.trk and a text file containing the list of labels
that should be part of the matrices. The ordering of labels in the matrices
will follow the same order as the list.

This script only generates matrices in the form of array, does not visualize
or reorder the labels (node).

The parameter --similarity expect a folder with density map (nii.gz) following
the same naming convention as the input directory.
They should the bundles average version in the same space. This will
compute the weigthed-dice between each node and their homologuous average
version.

The parameters --metrics can be used more than once and expect a map (t1, fa,
etc.) in the same space and each will generate a matrix. The average value in
the volume occupied by the bundle will be reported in the matrices nodes.

The parameters --maps can be used more than once and expect a folder with
pre-computed maps (nii.gz) following the same naming convention as the
input directory. Each will generate a matrix. The average non-zeros value in
the map will be reported in the matrices nodes.
"""

import argparse
import copy

from nibabel.streamlines import load, save, Tractogram
import numpy as np
from nibabel.streamlines.array_sequence import ArraySequence

from scilpy.io.streamlines import (load_tractogram_with_reference,
                                   save_from_voxel_space)
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.features import (remove_outliers,
                                           remove_loops_and_sharp_turns)
from scilpy.tractanalysis.tools import (compute_connectivity,
                                        compute_streamline_segment,
                                        extract_longest_segments_from_profile)
from scilpy.tractanalysis.uncompress import uncompress
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.io.streamlines import load_tractogram_with_reference


def _get_output_paths(args):
    root_dir = args.output_dir
    paths = {'raw': os.path.join(root_dir, 'raw_connections/'),
             'final': os.path.join(root_dir, 'final_connections/'),
             'removed_length': os.path.join(root_dir, 'removed_length/'),
             'loops': os.path.join(root_dir, 'loops/'),
             'outliers': os.path.join(root_dir, 'outliers/'),
             'qb_loops': os.path.join(root_dir, 'qb_loops/'),
             'pruned': os.path.join(root_dir, 'pruned/'),
             'no_loops': os.path.join(root_dir, 'no_loops/'),
             'no_outliers': os.path.join(root_dir, 'no_outliers/')}

    return paths


def _get_saving_options(args):
    saving_options = {'raw': args.save_raw_connections,
                      'intermediate': args.save_intermediate,
                      'discarded': args.save_discarded,
                      'final': True}

    return saving_options


def _create_required_output_dirs(args):
    out_paths = _get_output_paths(args)
    os.mkdir(out_paths['final'])

    if args.save_raw_connections:
        os.mkdir(out_paths['raw'])

    if args.save_discarded:
        os.mkdir(out_paths['loops'])
        os.mkdir(out_paths['outliers'])
        os.mkdir(out_paths['qb_loops'])
        os.mkdir(out_paths['removed_length'])

    if args.save_intermediate:
        os.mkdir(out_paths['pruned'])
        os.mkdir(out_paths['no_loops'])
        os.mkdir(out_paths['no_outliers'])


def _save_if_needed(sft, args, save_type, step_type, in_label, out_label):
    saving_options = _get_saving_options(args)
    out_paths = _get_output_paths(args)

    if saving_options[save_type] and len(sft):
        out_name = os.path.join(out_paths[step_type],
                                '{}_{}.trk'.format(in_label,
                                                   out_label))
        save_tractogram(sft, out_name, bbox_valid_check=False)


def _symmetrize_con_info(con_info):
    final_con_info = {}
    for in_label in list(con_info.keys()):
        for out_label in list(con_info[in_label].keys()):

            pair_info = con_info[in_label][out_label]

            final_in_label = min(in_label, out_label)
            final_out_label = max(in_label, out_label)

            if final_con_info.get(final_in_label) is None:
                final_con_info[final_in_label] = {}

            if final_con_info[final_in_label].get(final_out_label) is None:
                final_con_info[final_in_label][final_out_label] = []

            final_con_info[final_in_label][final_out_label].extend(pair_info)

    return final_con_info


def _prune_segments(segments, min_length, max_length, vox_size):
    lengths = list(length(segments) * vox_size)
    valid = []
    invalid = []

    for s, l in zip(segments, lengths):
        if min_length <= l <= max_length:
            valid.append(s)
        else:
            invalid.append(s)
    return valid, invalid


def processing_wrapper(args):
    args_from_parser = args[0]
    in_label, out_label = args[1]
    final_con_info = args[2]
    sft = args[3]
    indices, points_to_idx = args[4:6]
    vox_sizes = args[6]

    saving_options = _get_saving_options(args_from_parser)
    out_paths = _get_output_paths(args_from_parser)
    pair_info = final_con_info[in_label][out_label]
    streamlines = sft.streamlines

    if not len(pair_info):
        return

    final_strl = []

from scilpy.tracking.tools import resample_streamlines
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)

from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii
from builtins import zip
import argparse
import logging
import os
import time
import multiprocessing
import itertools
import multiprocessing
import logging
import os

import coloredlogs
from dipy.io.utils import is_header_compatible
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamlinespeed import length
import nibabel as nib
import numpy as np

from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.io.utils import (add_overwrite_arg, add_processes_args,
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
            logging.error('%s and %s do not have a compatible header',
                          in_filename, ref_filename)
            raise IOError
        return nib.load(in_filename).get_data()


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
    affine, dimensions, voxel_sizes, _ = sft.space_attribute
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
        density_sim = load_node_nifti(
            similarity_directory, in_label, out_label, in_filename)
        _, w_dice = compute_dice_voxel(density, density_sim)

        measures_to_return['similarity'] = w_dice
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
                logging.error('%s and %s do not have a compatible header',
                              in_filename, metric_filename)
                raise IOError

            metric_data = nib.load(metric_filename).get_data()
            if weighted:
                density = density / np.max(density)
                voxels_value = metric_data * density
                voxels_value = voxels_value[voxels_value > 0]
            else:
                voxels_value = metric_data[density > 0]

            measures_to_return[metric_filename] = np.average(voxels_value)

    return {(in_label, out_label): measures_to_return}


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_bundles_dir',
                   help='Folder containing all the bundle files (trk).')
    p.add_argument('labels_list',
                   help='Text file containing the list of labels from the '
                   'atlas.')
    p.add_argument('--volume', metavar='OUT_FILE',
                   help='Output file for the volume weighted matrix')
    p.add_argument('--streamline_count', metavar='OUT_FILE',
                   help='Output file for the streamline count weighted matrix.')
    p.add_argument('--length', metavar='OUT_FILE',
                   help='Output file for the length weighted matrix.')
    p.add_argument('--similarity', nargs=2,
                   metavar=('IN_FOLDER', 'OUT_FILE'),
                   help='Input folder containing the average bundles. \n'
                   'and output file for the similarity weigthed matrix.')
    p.add_argument('--maps', nargs=2,  action='append',
                   metavar=('IN_FOLDER', 'OUT_FILE'),
                   help='Input folder containing pre-computed maps. \n'
                   'and output file for that weigthed matrix.')
    p.add_argument('--metrics', nargs=2, action='append',
                   metavar=('IN_FILE', 'OUT_FILE'),
                   help='Input and output file for the metric weigthed matrix.')

    p.add_argument('--density_weigthing', action="store_true",
                   help='Use density-weighting for the metric weigthed matrix.')
    p.add_argument('--no_self_connection', action="store_true",
                   help='Eliminate the diagonal from the matrices.')

    add_overwrite_arg(p)
    add_processes_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()
    
    logging.info('*** Loading streamlines ***')
    time1 = time.time()
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    time2 = time.time()
    logging.info('    Loading %s streamlines took %0.2f sec.',
                 len(sft), (time2 - time1))

    logging.info('*** Filtering streamlines ***')
    data_mask = np.zeros(data_labels.shape)
    data_mask[data_labels > 0] = 1

    original_len = len(sft)
    time1 = time.time()
    sft.to_vox()
    sft.to_corner()
    sft.remove_invalid_streamlines()
    time2 = time.time()
    logging.info('    Discarded %s streamlines from filtering in %0.2f sec.',
                 original_len - len(sft), (time2 - time1))
    logging.info('    Number of streamlines to process: %s', len(sft))

    # Get all streamlines intersection indices
    logging.info('*** Computing streamlines intersection ***')
    time1 = time.time()

    indices, points_to_idx = uncompress(sft.streamlines, return_mapping=True)

    time2 = time.time()
    logging.info('    Streamlines intersection took %0.2f sec.',
                 (time2 - time1))

    # Compute the connectivity mapping
    logging.info('*** Computing connectivity information ***')
    time1 = time.time()
    con_info = compute_connectivity(indices, img_labels.get_data(),
                                    extract_longest_segments_from_profile)
    time2 = time.time()
    logging.info('    Connectivity computation took %0.2f sec.',
                 (time2 - time1))

    # Prepare directories and information needed to save.
    saving_opts = _get_saving_options(args)
    out_paths = _get_output_paths(args)
    _create_required_output_dirs(args)

    logging.info('*** Starting connection post-processing and saving. ***')
    logging.info('    This can be long, be patient.')
    time1 = time.time()
    real_labels = np.unique(img_labels.get_data())[1:]
    comb_list = list(itertools.combinations(real_labels, r=2))
    sft.to_rasmm()
    sft.to_center()
    
    pool = multiprocessing.Pool(args.nbr_processes)
    _ = pool.map(processing_wrapper,
                 zip(itertools.repeat(args),
                     comb_list,
                     itertools.repeat(con_info),
                     itertools.repeat(sft),
                     itertools.repeat(indices),
                     itertools.repeat(points_to_idx),
                     itertools.repeat(vox_sizes)))

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
                continue

            # This is necessary to support more than one map for weighting
            dict_metrics_out_name[in_name] = out_name
            measures_output_filename.append(out_name)

    assert_outputs_exist(parser, args, measures_output_filename)
    if not measures_to_compute:
        parser.error('No connectivity measures were selected, nothing'
                     'to compute.')
    logging.info('The following measures will be computed and save: %s',
                 measures_to_compute)

    labels_list = np.loadtxt(args.labels_list, dtype=int).tolist()

    comb_list = list(itertools.combinations(labels_list, r=2))
    if not args.no_self_connection:
        comb_list.extend(zip(labels_list, labels_list))

    pool = multiprocessing.Pool(args.nbr_processes)
    measures_dict_list = pool.map(_processing_wrapper,
                                  zip(itertools.repeat(args.in_bundles_dir),
                                      comb_list,
                                      itertools.repeat(measures_to_compute),
                                      itertools.repeat(args.density_weigthing),
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

    logging.info('Out of %s node, only %s contain values',
                 total_elem, len(measures_dict.keys())*2)

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
