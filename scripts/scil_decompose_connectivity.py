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
- 15 minutes without post-processing, only saving final bundles.
- 30 minutes with full post-processing, only saving final bundles.
- 60 minutes with full post-processing, saving all possible files.
"""

import argparse
import itertools
import logging
import os
import time

import coloredlogs
from dipy.io.stateful_tractogram import (Origin, Space,
                                         StatefulTractogram,
                                         set_sft_logger_level)
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamlinespeed import length
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.features import (remove_outliers,
                                           remove_loops_and_sharp_turns)
from scilpy.tractanalysis.tools import (compute_connectivity,
                                        compute_streamline_segment,
                                        extract_longest_segments_from_profile)
from scilpy.tractanalysis.uncompress import uncompress
from scilpy.io.streamlines import load_tractogram_with_reference


def _get_output_paths(args):
    root_dir = args.output_dir
    paths = {'raw': os.path.join(root_dir, 'raw_connections/'),
             'final': os.path.join(root_dir, 'final_connections/'),
             'invalid_length': os.path.join(root_dir, 'invalid_length/'),
             'valid_length': os.path.join(root_dir, 'valid_length/'),
             'loops': os.path.join(root_dir, 'loops/'),
             'outliers': os.path.join(root_dir, 'outliers/'),
             'qb_curv': os.path.join(root_dir, 'qb_curv/'),
             'no_loops': os.path.join(root_dir, 'no_loops/'),
             'inliers': os.path.join(root_dir, 'inliers/')}

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
        os.mkdir(out_paths['qb_curv'])
        os.mkdir(out_paths['invalid_length'])

    if args.save_intermediate:
        os.mkdir(out_paths['no_loops'])
        os.mkdir(out_paths['inliers'])
        os.mkdir(out_paths['valid_length'])


def _save_if_needed(streamlines, args,
                    save_type, step_type,
                    in_label, out_label):
    saving_options = _get_saving_options(args)
    out_paths = _get_output_paths(args)

    if saving_options[save_type] and len(streamlines):
        out_name = os.path.join(out_paths[step_type],
                                '{}_{}.trk'.format(in_label,
                                                   out_label))
        sft = StatefulTractogram(streamlines, args.in_tractogram,
                                 Space.VOX, origin=Origin.TRACKVIS)

        save_tractogram(sft, out_name)


def _prune_segments(segments, min_length, max_length, vox_size):
    lengths = list(length(segments) * vox_size)
    valid = []
    invalid = []

    for se, le in zip(segments, lengths):
        if min_length <= le <= max_length:
            valid.append(se)
        else:
            invalid.append(se)
    return valid, invalid


def build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_tractogram',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy.')
    p.add_argument('labels',
                   help='Labels file name (nifti).\nLabels must be consecutive '
                        'from 0 to N, with 0 the background.\n'
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
    og.add_argument('--outlier_threshold', type=float, default=0.6,
                    help='Outlier removal threshold when using hierarchical '
                         'QB. [%(default)s]')
    og.add_argument('--loop_max_angle', type=float, default=330.,
                    help='Maximal winding angle over which a streamline is '
                         'considered as looping. [%(default)s]')
    og.add_argument('--curv_qb_distance', type=float, default=10.,
                    help='Maximal distance to a centroid for curvature '
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

    p.add_argument('--out_labels_list', metavar='OUT_FILE',
                   help='Save the labels list as text file.\n'
                        'Needed for scil_compute_connectivity.py and others.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.labels])

    if os.path.abspath(args.output_dir) == os.getcwd():
        parser.error('Do not use the current path as output directory.')

    assert_output_dirs_exist_and_empty(parser, args, args.output_dir,
                                       create_dir=True)

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)
    coloredlogs.install(level=log_level)
    set_sft_logger_level('WARNING')

    img_labels = nib.load(args.labels)
    data_labels = img_labels.get_fdata().astype(np.int16)
    real_labels = np.unique(data_labels)[1:]
    if args.out_labels_list:
        np.savetxt(args.out_labels_list, real_labels, fmt='%i')

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
    logging.info('    Loading {} streamlines took {} sec.'.format(
        len(sft), round(time2 - time1, 2)))

    logging.info('*** Filtering streamlines ***')
    data_mask = np.zeros(data_labels.shape)
    data_mask[data_labels > 0] = 1

    original_len = len(sft)
    time1 = time.time()

    sft.to_vox()
    sft.to_corner()
    sft.remove_invalid_streamlines()
    time2 = time.time()
    logging.info(
        '    Discarded {} streamlines from filtering in {} sec.'.format(
          original_len - len(sft), round(time2 - time1, 2)))
    logging.info('    Number of streamlines to process: {}'.format(len(sft)))

    # Get all streamlines intersection indices
    logging.info('*** Computing streamlines intersection ***')
    time1 = time.time()

    indices, points_to_idx = uncompress(sft.streamlines, return_mapping=True)

    time2 = time.time()
    logging.info('    Streamlines intersection took {} sec.'.format(
        round(time2 - time1, 2)))

    # Compute the connectivity mapping
    logging.info('*** Computing connectivity information ***')
    time1 = time.time()
    con_info = compute_connectivity(indices,
                                    data_labels, real_labels,
                                    extract_longest_segments_from_profile)
    time2 = time.time()
    logging.info('    Connectivity computation took {} sec.'.format(
        round(time2 - time1, 2)))

    # Prepare directories and information needed to save.
    _create_required_output_dirs(args)

    logging.info('*** Starting connection post-processing and saving. ***')
    logging.info('    This can be long, be patient.')
    time1 = time.time()

    # Saving will be done from streamlines already in the right space
    comb_list = list(itertools.combinations(real_labels, r=2))

    iteration_counter = 0
    for in_label, out_label in comb_list:
        if iteration_counter > 0 and iteration_counter % 100 == 0:
            logging.info('Split {} nodes out of {}'.format(iteration_counter,
                                                           len(comb_list)))
        iteration_counter += 1

        pair_info = []
        if in_label not in con_info:
            continue
        elif out_label in con_info[in_label]:
            pair_info.extend(con_info[in_label][out_label])

        if out_label not in con_info:
            continue
        elif in_label in con_info[out_label]:
            pair_info.extend(con_info[out_label][in_label])

        if not len(pair_info):
            continue

        connecting_streamlines = []
        for connection in pair_info:
            strl_idx = connection['strl_idx']
            curr_streamlines = compute_streamline_segment(
                sft.streamlines[strl_idx],
                indices[strl_idx],
                connection['in_idx'],
                connection['out_idx'],
                points_to_idx[strl_idx])
            connecting_streamlines.append(curr_streamlines)

        _save_if_needed(connecting_streamlines, args, 'raw',
                        'raw', in_label, out_label)

        # Doing all post-processing
        if not args.no_pruning:
            valid_length, invalid_length = _prune_segments(
                connecting_streamlines,
                args.min_length,
                args.max_length,
                vox_sizes[0])

            _save_if_needed(invalid_length, args,
                            'discarded', 'invalid_length',
                            in_label, out_label)
        else:
            valid_length = connecting_streamlines

        if not len(valid_length):
            continue

        _save_if_needed(valid_length, args,
                        'intermediate', 'valid_length', in_label, out_label)

        if not args.no_remove_loops:
            no_loop_ids = remove_loops_and_sharp_turns(valid_length,
                                                       args.loop_max_angle)
            no_loops = [valid_length[i] for i in no_loop_ids]

            loop_ids = np.setdiff1d(np.arange(len(valid_length)), no_loop_ids)
            loops = [valid_length[i] for i in loop_ids]

            _save_if_needed(loops, args,
                            'discarded', 'loops', in_label, out_label)
        else:
            no_loops = valid_length

        if not len(no_loops):
            continue

        _save_if_needed(no_loops, args,
                        'intermediate', 'no_loops', in_label, out_label)

        if not args.no_remove_outliers:
            inliers, outliers = remove_outliers(no_loops,
                                                args.outlier_threshold)
            _save_if_needed(outliers, args,
                            'discarded', 'outliers', in_label, out_label)
        else:
            inliers = no_loops

        if not len(inliers):
            continue

        _save_if_needed(inliers, args,
                        'intermediate', 'inliers', in_label, out_label)

        if not args.no_remove_curv_dev:
            no_qb_curv_ids = remove_loops_and_sharp_turns(
                inliers,
                args.loop_max_angle,
                use_qb=True,
                qb_threshold=args.curv_qb_distance)
            no_qb_curv = [inliers[i] for i in no_qb_curv_ids]

            qb_curv_ids = np.setdiff1d(
                np.arange(len(inliers)), no_qb_curv_ids)
            qb_curv = [inliers[i] for i in qb_curv_ids]

            _save_if_needed(qb_curv, args,
                            'discarded', 'qb_curv', in_label, out_label)
        else:
            no_qb_curv = inliers

        _save_if_needed(no_qb_curv, args,
                        'final', 'final', in_label, out_label)

    time2 = time.time()
    logging.info(
        '    Connections post-processing and saving took {} sec.'.format(
          round(time2 - time1, 2)))


if __name__ == "__main__":
    main()
