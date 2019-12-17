#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

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


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='')
    p.add_argument('in_bundles', nargs='+'
                   help='Support tractography file')
    p.add_argument('json',
                   help='ordering')
    p.add_argument('--map',
                   help='For weigthed')
    p.add_argument('--volume', action="store_true",
                   help='')
    p.add_argument('--streamlines_count', action="store_true",
                   help='')
    p.add_argument('--length', action="store_true",
                   help='')
    p.add_argument('--similarity', action="store_true",
                   help='Support tractography file')

    add_reference_arg(p)
    add_overwrite_arg(p)

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



if __name__ == "__main__":
    main()
