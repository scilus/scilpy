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
"""

from __future__ import division

from builtins import zip
import argparse
import logging
import os
import time

from dipy.tracking.streamlinespeed import length
import nibabel as nb
import numpy as np

from scilpy.io.streamlines import (load_trk_in_voxel_space,
                                   save_from_voxel_space)
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.features import (remove_outliers,
                                           remove_loops_and_sharp_turns)
from scilpy.tractanalysis.tools import (compute_connectivity,
                                        compute_streamline_segment,
                                        extract_longest_segments_from_profile)
from scilpy.tractanalysis.uncompress import uncompress


def _get_output_paths(root_dir):
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


def _create_required_output_dirs(out_paths, args):
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


def _save_if_needed(streamlines, args, saving_options, out_paths,
                    save_type, step_type, in_label, out_label):
    if saving_options[save_type] and len(streamlines):
        save_from_voxel_space(streamlines, args.labels, args.tracks,
                              os.path.join(out_paths[step_type],
                                           '{}_{}.trk'.format(in_label,
                                                              out_label)))


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


def build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('tracks',
                   help='Path of the tracks file, in a format supported by ' +
                        'the Nibabel streamlines API.')
    p.add_argument('labels',
                   help='Labels file name (nifti). Labels must be consecutive '
                        'from 0 to N, with 0 the background. '
                        'This generates a NxN connectivity matrix.')
    p.add_argument('max_labels', type=int,
                   help='Maximal label value that could be present in the '
                        'parcellation. Used to generate matrices with the '
                        'same size for all subjects.')
    p.add_argument(dest='output', metavar='output_dir',
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
    post_proc.add_argument('--no_remove_loops_again', action='store_true',
                           help='If set, will NOT remove streamlines that '
                                'loop according to QuickBundles.\n'
                                'Threshold based on --loop_qb_distance.')

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
    og.add_argument('--loop_qb_distance', type=float, default=15.,
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
                        'subdirectories.\nIncludes loops, outliers and '
                        'qb_loops')

    add_overwrite_arg(p)

    p.add_argument('--verbose', '-v', dest='verbose',
                   action='store_true', default=False,
                   help='Verbose. [%(default)s]')
    return p


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.tracks, args.labels])

    if os.path.abspath(args.output) == os.getcwd():
        parser.error('Do not use the current path as output directory.')

    assert_output_dirs_exist_and_empty(parser, args, args.output)

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)

    img_labels = nb.load(args.labels)
    if not np.issubdtype(img_labels.get_data_dtype().type, np.integer):
        parser.error("Label image should contain integers for labels.")

    # Ensure that voxel size is isotropic. Currently, for speed considerations,
    # we take the length in voxel space and multiply by the voxel size. For
    # this to work correctly, voxel size must be isotropic.
    vox_sizes = img_labels.header.get_zooms()
    if not np.mean(vox_sizes) == vox_sizes[0]:
        parser.error('Labels must be isotropic')

    if np.min(img_labels.get_data()) < 0 or \
            np.max(img_labels.get_data()) > args.max_labels:
        parser.error('Invalid labels in labels image')

    logging.info('*** Loading streamlines ***')
    time1 = time.time()
    streamlines = load_trk_in_voxel_space(args.tracks, args.labels)
    time2 = time.time()

    logging.info('    Number of streamlines to process: {}'.format(
        len(streamlines)))
    logging.info('    Loading streamlines took %0.3f ms',
                 (time2 - time1) * 1000.0)

    # Get all streamlines intersection indices
    logging.info('*** Computing streamlines intersection ***')
    time1 = time.time()

    indices, points_to_idx = uncompress(streamlines, return_mapping=True)

    time2 = time.time()
    logging.info('    Streamlines intersection took %0.3f ms',
                 (time2 - time1) * 1000.0)

    # Compute the connectivity mapping
    # TODO self connection?
    logging.info('*** Computing connectivity information ***')
    time1 = time.time()
    con_info = compute_connectivity(indices, img_labels.get_data(),
                                    extract_longest_segments_from_profile,
                                    False, True)
    time2 = time.time()
    logging.info('    Connectivity computation took %0.3f ms',
                 (time2 - time1) * 1000.0)

    # Symmetrize matrix
    final_con_info = _symmetrize_con_info(con_info)

    # Prepare directories and information needed to save.
    saving_opts = _get_saving_options(args)
    out_paths = _get_output_paths(args.output)
    _create_required_output_dirs(out_paths, args)

    # Here, we use nb_labels + 1 since we want the direct mapping from image
    # label to matrix element. We will remove the first row and column before
    # saving.
    # TODO for other metrics
    # dtype should be adjusted depending on the type of elements
    # stored in the con_mat
    nb_labels = args.max_labels
    con_mat = np.zeros((nb_labels + 1, nb_labels + 1),
                       dtype=np.uint32)

    logging.info('*** Starting connection post-processing and saving. ***')
    logging.info('    This can be long, be patient.')
    time1 = time.time()
    for in_label in list(final_con_info.keys()):
        for out_label in list(final_con_info[in_label].keys()):
            pair_info = final_con_info[in_label][out_label]

            if not len(pair_info):
                continue

            final_strl = []

            for connection in pair_info:
                strl_idx = connection['strl_idx']
                final_strl.append(compute_streamline_segment(streamlines[strl_idx],
                                                             indices[strl_idx],
                                                             connection['in_idx'],
                                                             connection['out_idx'],
                                                             points_to_idx[strl_idx]))

            _save_if_needed(final_strl, args, saving_opts, out_paths, 'raw',
                            'raw', in_label, out_label)

            # Doing all post-processing
            if not args.no_pruning:
                pruned_strl, invalid_strl = _prune_segments(final_strl,
                                                            args.min_length,
                                                            args.max_length,
                                                            vox_sizes[0])

                _save_if_needed(invalid_strl, args, saving_opts, out_paths,
                                'discarded', 'removed_length',
                                in_label, out_label)
            else:
                pruned_strl = final_strl

            if not len(pruned_strl):
                continue

            _save_if_needed(pruned_strl, args, saving_opts, out_paths,
                            'intermediate', 'pruned', in_label, out_label)

            if not args.no_remove_loops:
                no_loops, loops = remove_loops_and_sharp_turns(pruned_strl,
                                                               args.loop_max_angle)
                _save_if_needed(loops, args, saving_opts, out_paths,
                                'discarded', 'loops', in_label, out_label)
            else:
                no_loops = pruned_strl

            if not len(no_loops):
                continue

            _save_if_needed(no_loops, args, saving_opts, out_paths,
                            'intermediate', 'no_loops', in_label, out_label)

            if not args.no_remove_outliers:
                no_outliers, outliers = remove_outliers(no_loops,
                                                        args.outlier_threshold)
                _save_if_needed(outliers, args, saving_opts, out_paths,
                                'discarded', 'outliers', in_label, out_label)
            else:
                no_outliers = no_loops

            if not len(no_outliers):
                continue

            _save_if_needed(no_outliers, args, saving_opts, out_paths,
                            'intermediate', 'no_outliers', in_label, out_label)

            if not args.no_remove_loops_again:
                no_qb_loops_strl, loops2 = remove_loops_and_sharp_turns(
                                                    no_outliers,
                                                    args.loop_max_angle,
                                                    True,
                                                    args.loop_qb_distance)
                _save_if_needed(loops2, args, saving_opts, out_paths,
                                'discarded', 'qb_loops', in_label, out_label)
            else:
                no_qb_loops_strl = no_outliers

            _save_if_needed(no_qb_loops_strl, args, saving_opts, out_paths,
                            'final', 'final', in_label, out_label)

            # TODO for other metrics
            # This would be where this is modified and the value
            # is computed (eg: mean FA in the connection.
            con_mat[in_label, out_label] += len(no_qb_loops_strl)

    time2 = time.time()
    logging.info('    Connection post-processing and saving took %0.3f ms',
                 (time2 - time1) * 1000.0)

    # Remove first line and column, since they are index 0 and
    # would represent a connection to non-label voxels. Only used when
    # post-processing to avoid unnecessary -1 on labels for each access.
    con_mat = con_mat[1:, 1:]
    np.save(os.path.join(args.output, 'final_matrix.npy'), con_mat)

    # In the future, could also be saved as a .json file, using pandas.
    # Contact JC Houde for example.


if __name__ == "__main__":
    main()
