#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute a connectivity matrix from a tractogram and a parcellation.

Current strategy is to keep the longest streamline segment connecting
2 regions. If the streamline crosses other gray matter regions before
reaching its final connected region, the kept connection is still the
longest.

This is robust to compressed streamlines.

NOTE: this script can take a while to run. Please be patient.
      Example: on a tractogram with 1.8M streamlines, running on a SSD:
               - 4 minutes without post-processing, only saving final bundles.
               - 29 minutes with full post-processing, only saving final bundles.
               - 30 minutes with full post-processing, saving all possible files.
"""

from __future__ import division

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
from dipy.io.streamline import save_tractogram
import nibabel as nb
import numpy as np
from nibabel.streamlines.array_sequence import ArraySequence

from scilpy.io.streamlines import (load_trk_in_voxel_space,
                                   save_from_voxel_space)
from scilpy.io.utils import (add_overwrite_arg, add_processes_args,
                             add_verbose_arg, add_reference_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.features import (remove_outliers,
                                           remove_loops_and_sharp_turns)
from scilpy.tractanalysis.tools import (compute_connectivity,
                                        compute_streamline_segment,
                                        extract_longest_segments_from_profile)
from scilpy.tractanalysis.uncompress import uncompress
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.io.streamlines import load_tractogram_with_reference, ichunk


global con_info, sft, indices, points_to_idx
from multiprocessing.managers import SharedMemoryManager


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


def _generated_saving_scheme(sft, args, save_type, step_type, in_label, out_label):
    saving_options = _get_saving_options(args)
    out_paths = _get_output_paths(args)

    if saving_options[save_type] and len(sft):
        out_name = os.path.join(out_paths[step_type],
                                '{}_{}.trk'.format(in_label,
                                                   out_label))
        return sft, out_name


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


def _processing_wrapper(args):
    args_from_parser = args[0]
    in_label, out_label = args[1]
    streamlines = args[2]
    final_con_info = args[3]
    vox_sizes = args[4]

    saving_options = _get_saving_options(args_from_parser)
    out_paths = _get_output_paths(args_from_parser)
    pair_info = final_con_info[in_label][out_label]
    # streamlines = sft.streamlines

    if not len(pair_info):
        return

    final_strl = []
    to_save_list = []
    for connection in pair_info:
        strl_idx = connection['strl_idx']
        final_strl.append(compute_streamline_segment(streamlines[strl_idx],
                                                     indices[strl_idx],
                                                     connection['in_idx'],
                                                     connection['out_idx'],
                                                     points_to_idx[strl_idx]))

    tmp_sft = StatefulTractogram(final_strl, sft, Space.RASMM)
    to_save = _generated_saving_scheme(tmp_sft, args_from_parser, 'raw',
                    'raw', in_label, out_label)
    to_save_list.append(to_save)

    # Doing all post-processing
    if not args_from_parser.no_pruning:
        pruned_strl, invalid_strl = _prune_segments(final_strl,
                                                    args_from_parser.min_length,
                                                    args_from_parser.max_length,
                                                    vox_sizes[0])

        tmp_sft = StatefulTractogram(invalid_strl, sft, Space.RASMM)
        to_save =_generated_saving_scheme(tmp_sft, args_from_parser,
                        'discarded', 'removed_length',
                        in_label, out_label)
        to_save_list.append(to_save)
    else:
        pruned_strl = final_strl

    if not len(pruned_strl):
        return

    tmp_sft = StatefulTractogram(pruned_strl, sft, Space.RASMM)
    to_save =_generated_saving_scheme(tmp_sft, args_from_parser,
                    'intermediate', 'pruned', in_label, out_label)
    to_save_list.append(to_save)

    if not args_from_parser.no_remove_loops:
        no_loops, loops = remove_loops_and_sharp_turns(pruned_strl,
                                                       args_from_parser.loop_max_angle)

        tmp_sft = StatefulTractogram(loops, sft, Space.RASMM)
        to_save =_generated_saving_scheme(tmp_sft, args_from_parser,
                        'discarded', 'loops', in_label, out_label)
        to_save_list.append(to_save)
    else:
        no_loops = pruned_strl

    if not len(no_loops):
        return

    tmp_sft = StatefulTractogram(no_loops, sft, Space.RASMM)
    to_save =_generated_saving_scheme(tmp_sft, args_from_parser,
                    'intermediate', 'no_loops', in_label, out_label)
    to_save_list.append(to_save)

    if not args_from_parser.no_remove_outliers:
        no_outliers, outliers = remove_outliers(no_loops,
                                                args_from_parser.outlier_threshold)

        tmp_sft = StatefulTractogram(outliers, sft, Space.RASMM)
        to_save =_generated_saving_scheme(tmp_sft, args_from_parser,
                        'discarded', 'outliers', in_label, out_label)
        to_save_list.append(to_save)
    else:
        no_outliers = no_loops

    if not len(no_outliers):
        return

    tmp_sft = StatefulTractogram(no_outliers, sft, Space.RASMM)
    to_save =_generated_saving_scheme(tmp_sft, args_from_parser,
                    'intermediate', 'no_outliers', in_label, out_label)
    to_save_list.append(to_save)

    if not args_from_parser.no_remove_curv_dev:
        no_qb_loops_strl, loops2 = remove_loops_and_sharp_turns(
            no_outliers,
            args_from_parser.loop_max_angle,
            True,
            args_from_parser.curv_qb_distance)

        tmp_sft = StatefulTractogram(loops2, sft, Space.RASMM)
        to_save =_generated_saving_scheme(tmp_sft, args_from_parser,
                        'discarded', 'qb_loops', in_label, out_label)
        to_save_list.append(to_save)
    else:
        no_qb_loops_strl = no_outliers

    tmp_sft = StatefulTractogram(no_qb_loops_strl, sft, Space.RASMM)
    to_save =_generated_saving_scheme(tmp_sft, args_from_parser,
                    'final', 'final', in_label, out_label)
    to_save_list.append(to_save)

    return to_save_list


def build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_tractogram',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy.')
    p.add_argument('labels',
                   help='Labels file name (nifti). Labels must be consecutive '
                        'from 0 to N, with 0 the background. '
                        'This generates a NxN connectivity matrix.')
    p.add_argument('output_dir',
                   help='Output directory path.')

    post_proc = p.add_argument_group('Post-processing options')
    post_proc.add_argument('--no_pruning', action='store_true',
                           help='If set, will NOT prune on length.\n'
                                'Length criteria in --min_length, '
                                '--max_length')
    post_proc.add_argument('--no_remove_loops', action='store_true',
                           help='If set, will NOT remove streamlines making '
                                'loops.\nAngle criteria based on '
                                '--loop_max_angle')
    post_proc.add_argument('--no_remove_outliers', action='store_true',
                           help='If set, will NOT remove outliers using QB.\n'
                                'Criteria based on --outlier_threshold.')
    post_proc.add_argument('--no_remove_curv_dev', action='store_true',
                           help='If set, will NOT remove streamlines that '
                                'deviate from the mean curvature.\n'
                                'Threshold based on --curv_qb_distance.')

    pr = p.add_argument_group('Pruning options')
    pr.add_argument('--min_length', type=float, default=20.,
                    help='Pruning minimal segment length. [%(default)s]')
    pr.add_argument('--max_length', type=float, default=200.,
                    help='Pruning maximal segment length. [%(default)s]')

    og = p.add_argument_group('Outliers and loops options')
    og.add_argument('--outlier_threshold', type=float, default=0.3,
                    help='Outlier removal threshold when using hierarchical '
                         'QB. [%(default)s]')
    og.add_argument('--loop_max_angle', type=float, default=360.,
                    help='Maximal winding angle over which a streamline is '
                         'considered as looping. [%(default)s]')
    og.add_argument('--curv_qb_distance', type=float, default=10.,
                    help='Maximal distance to a centroid for loop / turn '
                         'filtering with QB. [%(default)s]')

    s = p.add_argument_group('Saving options')
    s.add_argument('--save_raw_connections', action='store_true',
                   help='If set, will save all raw cut connections in a '
                        'subdirectory')
    s.add_argument('--save_intermediate', action='store_true',
                   help='If set, will save the intermediate results of '
                        'filtering')
    s.add_argument('--save_discarded', action='store_true',
                   help='If set, will save discarded streamlines in '
                        'subdirectories.\n'
                        'Includes loops, outliers and qb_loops')

    add_overwrite_arg(p)
    add_processes_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.labels])

    if os.path.abspath(args.output_dir) == os.getcwd():
        parser.error('Do not use the current path as output directory.')

    assert_output_dirs_exist_and_empty(parser, args, args.output_dir)
    global sft, indices, points_to_idx

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)
    coloredlogs.install(level=log_level)

    img_labels = nb.load(args.labels)
    data_labels = img_labels.get_data()
    if not np.issubdtype(img_labels.get_data_dtype().type, np.integer):
        parser.error("Label image should contain integers for labels.")

    # Voxel size must be isotropic, for speed/performance considerations
    vox_sizes = img_labels.header.get_zooms()
    if not np.mean(vox_sizes) == vox_sizes[0]:
        parser.error('Labels must be isotropic')

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

    step = int(len(sft) / args.nbr_processes + 0.5) + \
        len(sft) % args.nbr_processes
    chunks = ichunk(indices, step)
    chunks_size = range(0, len(sft), step)
    pool = multiprocessing.Pool(args.nbr_processes)

    con_info_list = pool.map(compute_connectivity,
                             zip(chunks,
                                 chunks_size,
                                 itertools.repeat(data_labels),
                                 itertools.repeat(extract_longest_segments_from_profile)))

    # Re-assemble the dictionary
    con_info = dict(con_info_list[0])
    for dix in con_info_list[1:]:
        for key_1 in dix.keys():
            for key_2 in dix[key_1].keys():
                con_info[key_1][key_2].extend(dix[key_1][key_2])
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
    real_labels = np.unique(data_labels)[1:]
    comb_list = list(itertools.combinations(real_labels, r=2))
    sft.to_rasmm()
    sft.to_center()

    smm = SharedMemoryManager()
    smm.start()
    sl = smm.ShareableList(sft.streamlines)

    to_save_lists = pool.map(_processing_wrapper,
                 zip(itertools.repeat(args),
                     comb_list,
                     itertools.repeat(sl),
                     itertools.repeat(con_info),
                     itertools.repeat(vox_sizes)))
    
    for to_save_list in to_save_lists:
        for to_save in to_save_list:
            if not None:
                save_tractogram(to_save[0], to_save[1])


    time2 = time.time()
    logging.info('    Connections post-processing and saving took %0.2f sec.',
                 (time2 - time1))


if __name__ == "__main__":
    main()
