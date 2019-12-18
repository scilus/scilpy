#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
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

import coloredlogs
from dipy.tracking.streamlinespeed import length
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram
import nibabel as nb
import numpy as np
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map

from scilpy.io.utils import (add_overwrite_arg, add_processes_args,
                             add_verbose_arg, add_reference_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)


def compute_dice_voxel(density_1, density_2):
    """
    Compute the overlap (dice coefficient) between two density maps (or binary).
    Parameters
    ----------
    density_1: ndarray
        Density (or binary) map computed from the first bundle
    density_1: ndarray of ndarray
        Density (or binary) map computed from the second bundle
    Returns
    -------
    A tuple containing
        float: Value between 0 and 1 that represent the spatial aggrement
            between both bundles.
        float: Value between 0 and 1 that represent the spatial aggrement
            between both bundles, weighted by streamlines density.
    """
    binary_1 = copy.copy(density_1)
    binary_1[binary_1 > 0] = 1
    binary_2 = copy.copy(density_2)
    binary_2[binary_2 > 0] = 1

    numerator = 2 * np.count_nonzero(binary_1 * binary_2)
    denominator = np.count_nonzero(binary_1) + np.count_nonzero(binary_2)
    if denominator > 0:
        dice = numerator / float(denominator)
    else:
        dice = np.nan

    indices = np.nonzero(binary_1 * binary_2)
    overlap_1 = density_1[indices]
    overlap_2 = density_2[indices]
    w_dice = (np.sum(overlap_1) + np.sum(overlap_2))
    denominator = float(np.sum(density_1) + np.sum(density_2))
    if denominator > 0:
        w_dice /= denominator
    else:
        w_dice = np.nan

    return dice, w_dice


def _processing_wrapper(args):
    bundles_dir = args[0]
    in_label, out_label = args[1]
    measures_to_compute = args[2]
    dict_map = args[3]
    weighted = args[4]
    similarity_directory = args[5]

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
        mean_length = np.average(length(sft.streamlines))

    sft.to_vox()
    sft.to_corner()
    density = compute_tract_counts_map(sft.streamlines,
                                       dimensions)

    if 'volume' in measures_to_compute:
        measures_to_return['volume'] = np.count_nonzero(density) * \
            np.prod(voxel_sizes)
        measures_to_compute.remove('volume')
    if 'streamlines_count' in measures_to_compute:
        measures_to_return['streamlines_count'] = len(sft)
        measures_to_compute.remove('streamlines_count')
    if 'length' in measures_to_compute:
        measures_to_return['length'] = mean_length
        measures_to_compute.remove('length')
    if 'similarity' in measures_to_compute and similarity_directory:
        in_filename_1 = os.path.join(similarity_directory,
                                     '{}_{}.trk'.format(in_label, out_label))
        in_filename_2 = os.path.join(similarity_directory,
                                     '{}_{}.trk'.format(out_label, in_label))
        in_filename_sim = None
        if os.path.isfile(in_filename_1):
            in_filename_sim = in_filename_1
        elif os.path.isfile(in_filename_2):
            in_filename_sim = in_filename_2


        if not in_filename_sim is None and is_header_compatible(in_filename_sim, in_filename):
            sft_sim = load_tractogram(in_filename_sim, 'same')
            _, dimensions, _, _ = sft.space_attribute

            sft_sim.to_vox()
            sft_sim.to_corner()
            density_sim = compute_tract_counts_map(sft_sim.streamlines,
                                                   dimensions)

            _, w_dice = compute_dice_voxel(density, density_sim)
            measures_to_return['similarity'] = w_dice
            measures_to_compute.remove('similarity')

    for map_base_name in measures_to_compute:
        if weighted:
            density = density / np.max(density)
            voxels_value = dict_map[map_base_name] * density
            voxels_value = voxels_value[voxels_value > 0]
        else:
            voxels_value = dict_map[map_base_name][density > 0]
            

        measures_to_return[map_base_name] = np.average(voxels_value)
        measures_to_compute.remove(map_base_name)

        return {(in_label, out_label): measures_to_return}

def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='')
    p.add_argument('in_bundles_dir',
                   help='Support tractography file')
    p.add_argument('labels_list',
                   help='ordering')
    p.add_argument('--volume', action="store_true",
                   help='')
    p.add_argument('--streamlines_count', action="store_true",
                   help='')
    p.add_argument('--length', action="store_true",
                   help='')
    p.add_argument('--similarity',
                   help='Support tractography file')
    p.add_argument('--maps', nargs='+',
                   help='For weigthed')
    p.add_argument('--density_weigth', action="store_true",
                   help='For weigthed')

    add_reference_arg(p)
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
    logging.info('    Loading %s streamlines took %0.2f ms',
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
    logging.info('    Discarded %s streamlines from filtering in %0.2f ms',
                 original_len - len(sft), (time2 - time1))
    logging.info('    Number of streamlines to process: %s', len(sft))

    # Get all streamlines intersection indices
    logging.info('*** Computing streamlines intersection ***')
    time1 = time.time()

    indices, points_to_idx = uncompress(sft.streamlines, return_mapping=True)

    time2 = time.time()
    logging.info('    Streamlines intersection took %0.2f ms',
                 (time2 - time1))

    # Compute the connectivity mapping
    logging.info('*** Computing connectivity information ***')
    time1 = time.time()
    con_info = compute_connectivity(indices, img_labels.get_data(),
                                    extract_longest_segments_from_profile)
    time2 = time.time()
    logging.info('    Connectivity computation took %0.2f ms',
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

    labels_list = np.loadtxt(args.labels_list, dtype=int)

    measures_to_compute = []

    if args.volume:
        measures_to_compute.append('volume')
    if args.streamlines_count:
        measures_to_compute.append('streamlines_count')
    if args.length:
        measures_to_compute.append('length')
    if args.similarity:
        measures_to_compute.append('similarity')

    dict_map = {}
    for filepath in args.maps:
        if not is_header_compatible(args.maps[0], filepath):
            continue
        base_name = os.path.basename(filepath)
        basename, _ = split_name_with_nii(base_name)
        measures_to_compute.append(base_name)
        map_data = nib.load(filepath).get_data()
        dict_map[base_name] = map_data

    comb_list = list(itertools.combinations(labels_list, r=2))

    pool = multiprocessing.Pool(args.nbr_processes)
    _ = pool.map(_processing_wrapper,
                 zip(itertools.repeat(args.in_bundles_dir),
                     comb_list,
                     itertools.repeat(measures_to_compute),
                     itertools.repeat(dict_map),
                     itertools.repeat(args.density_weigth),
                     itertools.repeat(args.similarity)))


if __name__ == "__main__":
    main()
