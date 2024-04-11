#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a connectivity matrix from a tractogram and a parcellation.

Current strategy is to keep the longest streamline segment connecting 2
regions. If the streamline crosses other gray matter regions before reaching
its final connected region, the kept connection is still the longest. This is
robust to compressed streamlines.

The output file is a hdf5 (.h5) where the keys are 'LABEL1_LABEL2' and each
group is composed of 'data', 'offsets' and 'lengths' from the array_sequence.
The 'data' is stored in VOX/CORNER for simplicity and efficiency. See script
scil_tractogram_convert_hdf5_to_trk.py to convert to a list of .trk bundles.

For the --outlier_threshold option the default is a recommended good trade-off
for a freesurfer parcellation. With smaller parcels (brainnetome, glasser) the
threshold should most likely be reduced.

Good candidate connections to QC are the brainstem to precentral gyrus
connection and precentral left to precentral right connection, or equivalent
in your parcellation.

NOTE: this script can take a while to run. Please be patient.
Example: on a tractogram with 1.8M streamlines, running on a SSD:
- 15 minutes without post-processing, only saving final bundles.
- 30 minutes with full post-processing, only saving final bundles.
- 60 minutes with full post-processing, saving all possible files.

Formerly: scil_decompose_connectivity.py
"""

import argparse
import itertools
import logging
import os
import time

import coloredlogs
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import get_reference_info, is_header_compatible
from dipy.tracking.streamlinespeed import length
import h5py
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.io.hdf5 import (construct_hdf5_header,
                            construct_hdf5_group_from_streamlines)
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_bbox_arg, add_overwrite_arg,
                             add_processes_arg, add_verbose_arg,
                             add_reference_arg, assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             validate_nbr_processes)
from scilpy.tractanalysis.bundle_operations import remove_outliers
from scilpy.tractanalysis.tools import (compute_connectivity,
                                        extract_longest_segments_from_profile)
from scilpy.tractograms.uncompress import uncompress
from scilpy.tractograms.streamline_operations import \
    remove_loops_and_sharp_turns
from scilpy.tractograms.streamline_and_mask_operations import \
    compute_streamline_segment


def _get_output_paths(args):
    root_dir = args.out_dir
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
    if not args.out_dir:
        return
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


def _save_if_needed(sft, hdf5_file, args,
                    save_type, step_type,
                    in_label, out_label):
    if step_type == 'final':
        # Due to the cutting, streamlines can become invalid
        indices = []
        for i in range(len(sft)):
            norm = np.linalg.norm(np.gradient(sft.streamlines[i],
                                              axis=0), axis=1)
            if (norm < 0.001).any():  # or len(sft.streamlines[i]) <= 1:
                indices.append(i)

        indices = np.setdiff1d(range(len(sft)), indices).astype(np.uint32)
        sft = sft[indices]

        group = hdf5_file.create_group('{}_{}'.format(in_label, out_label))
        construct_hdf5_group_from_streamlines(group, sft.streamlines,
                                              dps=sft.data_per_streamline)

    if args.out_dir:
        saving_options = _get_saving_options(args)
        out_paths = _get_output_paths(args)

        if saving_options[save_type] and len(sft):
            out_name = os.path.join(out_paths[step_type],
                                    '{}_{}.trk'.format(in_label,
                                                       out_label))
            save_tractogram(sft, out_name)


def _prune_segments(segments, min_length, max_length, vox_size):
    lengths = list(length(segments) * vox_size)
    valid = []
    invalid = []

    for i, tuple_zip in enumerate(zip(segments, lengths)):
        _, le = tuple_zip
        if min_length <= le <= max_length:
            valid.append(i)
        else:
            invalid.append(i)
    return valid, invalid


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_tractograms', nargs='+',
                   help='Tractogram filenames. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy.')
    p.add_argument('in_labels',
                   help='Labels file name (nifti). Labels must have 0 as '
                        'background.')
    p.add_argument('out_hdf5',
                   help='Output hdf5 file (.h5).')

    post_proc = p.add_argument_group('Post-processing options')
    post_proc.add_argument('--no_pruning', action='store_true',
                           help='If set, will NOT prune on length.\n'
                                'Length criteria in --min_length, '
                                '--max_length.')
    post_proc.add_argument('--no_remove_loops', action='store_true',
                           help='If set, will NOT remove streamlines making '
                                'loops.\nAngle criteria based on '
                                '--loop_max_angle.')
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
                    help='Clustering threshold for centroids curvature '
                         'filtering with QB. [%(default)s]')

    s = p.add_argument_group('Saving options')
    s.add_argument('--out_dir',
                   help='Output directory for each connection as separate '
                        'file (.trk).')
    s.add_argument('--save_raw_connections', action='store_true',
                   help='If set, will save all raw cut connections in a '
                        'subdirectory.')
    s.add_argument('--save_intermediate', action='store_true',
                   help='If set, will save the intermediate results of '
                        'filtering.')
    s.add_argument('--save_discarded', action='store_true',
                   help='If set, will save discarded streamlines in '
                        'subdirectories.\n'
                        'Includes loops, outliers and qb_loops.')

    p.add_argument('--out_labels_list', metavar='OUT_FILE',
                   help='Save the labels list as text file.\n'
                        'Needed for scil_connectivity_compute_matrices.py and '
                        'others.')

    add_reference_arg(p)
    add_bbox_arg(p)
    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    coloredlogs.install(level=logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractograms+[args.in_labels],
                        args.reference)
    assert_outputs_exist(parser, args, args.out_hdf5)
    nbr_cpu = validate_nbr_processes(parser, args)

    # HDF5 will not overwrite the file
    if os.path.isfile(args.out_hdf5):
        os.remove(args.out_hdf5)

    if (args.save_raw_connections or args.save_intermediate
            or args.save_discarded) and not args.out_dir:
        parser.error('To save outputs in the streamlines form, provide the '
                     'output directory using --out_dir.')

    if args.out_dir:
        if os.path.abspath(args.out_dir) == os.getcwd():
            parser.error('Do not use the current path as output directory.')
        assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                           create_dir=True)

    # Load everything
    img_labels = nib.load(args.in_labels)
    data_labels = get_data_as_labels(img_labels)
    real_labels = np.unique(data_labels)[1:]   # Removing the background 0.
    if args.out_labels_list:
        np.savetxt(args.out_labels_list, real_labels, fmt='%i')

    # Voxel size must be isotropic, for speed/performance considerations
    vox_sizes = img_labels.header.get_zooms()
    if not np.allclose(np.mean(vox_sizes), vox_sizes, atol=1e-01):
        parser.error('Labels must be isotropic')

    logging.info('*** Loading streamlines ***')
    time1 = time.time()
    sft = None
    for in_tractogram in args.in_tractograms:
        if sft is None:
            sft = load_tractogram_with_reference(parser, args, in_tractogram)
            if not is_header_compatible(sft, img_labels):
                raise IOError('{} and {} do not have a compatible '
                              'header'.format(in_tractogram, args.in_labels))
        else:
            sft += load_tractogram_with_reference(parser, args, in_tractogram)

    time2 = time.time()
    logging.info('    Loading {} streamlines took {} sec.'.format(
        len(sft), round(time2 - time1, 2)))

    sft.to_vox()
    sft.to_corner()

    # Get the indices of the voxels traversed by each streamline
    logging.info('*** Computing voxels traversed by each streamline ***')
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
    comb_list.extend(zip(real_labels, real_labels))

    iteration_counter = 0
    with h5py.File(args.out_hdf5, 'w') as hdf5_file:
        construct_hdf5_header(hdf5_file, sft)

        # Each connection is processed independently. Multiprocessing would be
        # a burden on the I/O of most SSD/HD
        for in_label, out_label in comb_list:
            if iteration_counter > 0 and iteration_counter % 100 == 0:
                logging.info('Split {} nodes out of {}'.format(
                    iteration_counter, len(comb_list)))
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
            connecting_ids = []
            for connection in pair_info:
                strl_idx = connection['strl_idx']
                curr_streamlines = compute_streamline_segment(
                    sft.streamlines[strl_idx],
                    indices[strl_idx],
                    connection['in_idx'],
                    connection['out_idx'],
                    points_to_idx[strl_idx])
                connecting_streamlines.append(curr_streamlines)
                connecting_ids.append(strl_idx)

            # Each step is processed from the previous 'success'
            #   1. raw         -> length pass/fail
            #   2. length pass -> loops pass/fail
            #   3. loops pass  -> outlier detection pass/fail
            #   4. outlier detection pass -> qb curvature pass/fail
            #   5. qb curvature pass == final connections
            connecting_streamlines = ArraySequence(connecting_streamlines)
            raw_dps = sft.data_per_streamline[connecting_ids]
            raw_sft = StatefulTractogram.from_sft(connecting_streamlines, sft,
                                                  data_per_streamline=raw_dps,
                                                  data_per_point={})
            _save_if_needed(raw_sft, hdf5_file, args,
                            'raw', 'raw', in_label, out_label)

            # Doing all post-processing
            if not args.no_pruning:
                valid_length_ids, invalid_length_ids = _prune_segments(
                    raw_sft.streamlines,
                    args.min_length,
                    args.max_length,
                    vox_sizes[0])

                invalid_length_sft = raw_sft[invalid_length_ids]
                valid_length = connecting_streamlines[valid_length_ids]
                _save_if_needed(invalid_length_sft, hdf5_file, args,
                                'discarded', 'invalid_length',
                                in_label, out_label)
            else:
                valid_length = connecting_streamlines
                valid_length_ids = range(len(connecting_streamlines))

            if not len(valid_length):
                continue

            valid_length_sft = raw_sft[valid_length_ids]
            _save_if_needed(valid_length_sft, hdf5_file, args,
                            'intermediate', 'valid_length', in_label,
                            out_label)

            if not args.no_remove_loops:
                no_loop_ids = remove_loops_and_sharp_turns(
                    valid_length,
                    args.loop_max_angle,
                    num_processes=nbr_cpu)
                loop_ids = np.setdiff1d(np.arange(len(valid_length)),
                                        no_loop_ids)

                loops_sft = valid_length_sft[loop_ids]
                no_loops = valid_length[no_loop_ids]
                _save_if_needed(loops_sft, hdf5_file, args,
                                'discarded', 'loops', in_label, out_label)
            else:
                no_loops = valid_length
                no_loop_ids = range(len(valid_length))

            if not len(no_loops):
                continue
            no_loops_sft = valid_length_sft[no_loop_ids]
            _save_if_needed(no_loops_sft, hdf5_file, args,
                            'intermediate', 'no_loops', in_label, out_label)

            if not args.no_remove_outliers:
                outliers_ids, inliers_ids = remove_outliers(
                    no_loops,
                    args.outlier_threshold,
                    nb_samplings=10,
                    fast_approx=True)

                outliers_sft = no_loops_sft[outliers_ids]
                inliers = no_loops[inliers_ids]
                _save_if_needed(outliers_sft, hdf5_file, args,
                                'discarded', 'outliers', in_label, out_label)
            else:
                inliers = no_loops
                inliers_ids = range(len(no_loops))

            if not len(inliers):
                continue

            inliers_sft = no_loops_sft[inliers_ids]
            _save_if_needed(inliers_sft, hdf5_file, args,
                            'intermediate', 'inliers', in_label, out_label)

            if not args.no_remove_curv_dev:
                no_qb_curv_ids = remove_loops_and_sharp_turns(
                    inliers,
                    args.loop_max_angle,
                    qb_threshold=args.curv_qb_distance,
                    num_processes=nbr_cpu)
                qb_curv_ids = np.setdiff1d(np.arange(len(inliers)),
                                           no_qb_curv_ids)

                qb_curv_sft = inliers_sft[qb_curv_ids]
                _save_if_needed(qb_curv_sft, hdf5_file, args,
                                'discarded', 'qb_curv', in_label, out_label)
            else:
                no_qb_curv_ids = range(len(inliers))

            no_qb_curv_sft = inliers_sft[no_qb_curv_ids]
            _save_if_needed(no_qb_curv_sft, hdf5_file, args,
                            'final', 'final', in_label, out_label)

    time2 = time.time()
    logging.info(
        '    Connections post-processing and saving took {} sec.'.format(
            round(time2 - time1, 2)))


if __name__ == "__main__":
    main()
