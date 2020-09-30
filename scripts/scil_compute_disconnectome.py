#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a connectivity matrix from a tractogram and a parcellation.

Current strategy is to keep the longest streamline segment connecting
2 regions. If the streamline crosses other gray matter regions before
reaching its final connected region, the kept connection is still the
longest. This is robust to compressed streamlines.

The output file is a hdf5 (.h5) where the keys are 'LABEL1_LABEL2' and each
group is composed of 'data', 'offsets' and 'lengths' from the array_sequence.
The 'data' is stored in VOX/CORNER for simplicity and efficiency.

For the --outlier_threshold option the default is a recommended good trade-off
for a freesurfer parcellation. With smaller parcels (brainnetome, glasser) the
threshold should most likely be reduced.
Good candidate connections to QC are the brainstem to precentral gyrus
connection and precentral left to precentral right connection, or equivalent
in your parcellation."

NOTE: this script can take a while to run. Please be patient.
Example: on a tractogram with 1.8M streamlines, running on a SSD:
- 15 minutes without post-processing, only saving final bundles.
- 30 minutes with full post-processing, only saving final bundles.
- 60 minutes with full post-processing, saving all possible files.
"""

import argparse
import itertools
import json
import logging
import os
import time

import coloredlogs
from dipy.io.stateful_tractogram import set_sft_logger_level
from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_label, get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             add_reference_arg,
                             add_json_args,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.segment.streamlines import filter_grid_roi


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_tractograms', nargs='+',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy.')
    p.add_argument('in_labels',
                   help='Labels file name (nifti). Labels must have 0 as '
                        'background.')
    p.add_argument('in_lesion',
                   help='Input binary mask representing the lesion(s).')

    s = p.add_argument_group('Saving options')
    p.add_argument('--out_json',
                   help='Output json containing all inputs as keys,\n'
                        'then diff/before/after keys and then a disconnectome '
                        'matrix')
    s.add_argument('--out_dir',
                   help='Save all output disconnectome matrices to this '
                        'directory.')
    s.add_argument('--save_diff_array', action='store_true',
                   help='Difference of streamlines count created by the '
                        'lesion(s).')
    s.add_argument('--save_before_array', action='store_true',
                   help='Equivalent to a streamline count matrix.')
    s.add_argument('--save_after_array', action='store_true',
                   help='Equivalent to a streamline count matrix minus'
                        'the streamlines touching the lesion(s).')

    p.add_argument('--force_labels_list', metavar='IN_TXT',
                   help='Path to a labels list (.txt) in case of missing '
                        'labels in the atlas.')

    add_json_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p

# List of TODO for the 'far future'
# TODO Supports a volume overlap with the lesion(s) matrix
# TODO Support a probability/weigthed count (peripheries/cores of lesions)
# TODO Split lesions by clusters and do a lesions count matrix
def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractograms+[args.in_labels,
                                                     args.in_lesion],
                        optional=args.force_labels_list)
    assert_outputs_exist(parser, args, [], optional=args.out_json)

    if args.out_dir:
        if os.path.abspath(args.out_dir) == os.getcwd():
            parser.error('Do not use the current path as output directory.')
        assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                           create_dir=True)
    elif args.save_diff_array or args.save_before_array or \
            args.save_after_array:
        parser.error('You must specificy a output directory to save arrays.')

    if not args.out_json and not args.save_diff_array and \
            not args.save_before_array and not args.save_after_array:
        parser.error('Choose at least one output options.')

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)
    coloredlogs.install(level=log_level)
    set_sft_logger_level('WARNING')

    img_labels = nib.load(args.in_labels)
    data_labels = get_data_as_label(img_labels)
    if args.force_labels_list:
        real_labels = np.loadtxt(args.force_labels_list,
                                 dtype=np.uint16)
    else:
        real_labels = np.unique(data_labels)[1:]
    nbr_labels = len(real_labels)

    lesions_data = get_data_as_mask(nib.load(args.in_lesion))

    # Voxel size must be isotropic, for speed/performance considerations
    vox_sizes = img_labels.header.get_zooms()
    if not np.allclose(np.mean(vox_sizes), vox_sizes, atol=1e-03):
        parser.error('Labels must be isotropic')

    results_dict = {}
    for filename in args.in_tractograms:
        logging.info('*** Loading streamlines from {} ***'.format(filename))
        time1 = time.time()
        sft = load_tractogram_with_reference(parser, args, filename)
        time2 = time.time()
        logging.info('    Loading {} streamlines took {} sec.'.format(
            len(sft), round(time2 - time1, 2)))

        if not is_header_compatible(sft, img_labels):
            raise IOError('{} and {} do not have a compatible header'.format(
                filename, args.in_labels))

        sft.to_vox()
        sft.to_corner()

        # Saving will be done from streamlines already in the right space
        comb_list = list(itertools.combinations(real_labels, r=2))
        comb_list.extend(zip(real_labels, real_labels))

        results_dict[filename] = {}
        results_dict[filename]['before'] = np.zeros((nbr_labels, nbr_labels),
                                                    dtype=np.uint32)
        results_dict[filename]['after'] = np.zeros((nbr_labels, nbr_labels),
                                                   dtype=np.uint32)
        results_dict[filename]['diff'] = np.zeros((nbr_labels, nbr_labels),
                                                  dtype=np.uint32)

        iteration_counter = 0
        for key_1 in real_labels:
            pos_1 = np.argwhere(real_labels == key_1)[0][0]
            data_1 = np.zeros(data_labels.shape, dtype=np.uint8)
            data_1[data_labels == key_1] = 1

            f_sft_1, _ = filter_grid_roi(sft, data_1, 'either_end',
                                         False)
            for key_2 in real_labels:
                if (key_1, key_2) not in comb_list:
                    continue
                iteration_counter += 1
                if iteration_counter > 0 and iteration_counter % 100 == 0:
                    logging.info('Computed {} nodes out of {}'.format(
                        iteration_counter, len(comb_list)))

                if len(f_sft_1) == 0:
                    continue

                pos_2 = np.argwhere(real_labels == key_2)[0][0]
                data_2 = np.zeros(data_labels.shape, dtype=np.uint8)
                data_2[data_labels == key_2] = 1

                mode = 'both_ends' if key_1 == key_2 else 'either_end'
                f_sft_2, _ = filter_grid_roi(f_sft_1, data_2, mode, False)

                results_dict[filename]['before'][pos_1, pos_2] = len(f_sft_2)
                f_sft, _ = filter_grid_roi(f_sft_2, lesions_data, 'any', True)
                results_dict[filename]['after'][pos_1, pos_2] = len(f_sft)

        # Symmetrize the disconnectome matrix
        results_dict[filename]['before'] += results_dict[filename]['before'].T
        results_dict[filename]['before'] -= (np.eye(len(real_labels),
                                                    dtype=np.uint32) *
                                             results_dict[filename]['before'])
        results_dict[filename]['after'] += results_dict[filename]['after'].T
        results_dict[filename]['after'] -= (np.eye(len(real_labels),
                                                   dtype=np.uint32) *
                                            results_dict[filename]['after'])

        # Generate the diff
        results_dict[filename]['diff'] = results_dict[filename]['before'] - \
            results_dict[filename]['after']

        # Convert to list for json
        results_dict[filename]['before'] = results_dict[filename]['before'].tolist()
        results_dict[filename]['after'] = results_dict[filename]['after'].tolist()
        results_dict[filename]['diff'] = results_dict[filename]['diff'].tolist()

        if args.out_dir:
            base, _ = os.path.splitext(filename)
            if args.save_before_array:
                np.save(os.path.join(args.out_dir,
                                     '{}_before.{}'.format(base, 'npy')),
                        results_dict[filename]['before'])
            if args.save_after_array:
                np.save(os.path.join(args.out_dir,
                                     '{}_after.{}'.format(base, 'npy')),
                        results_dict[filename]['after'])
            if args.save_diff_array:
                np.save(os.path.join(args.out_dir,
                                     '{}_diff.{}'.format(base, 'npy')),
                        results_dict[filename]['diff'])

    if args.out_json:
        with open(args.out_json, 'w') as out_file:
            json.dump(results_dict, out_file, indent=args.indent,
                      sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
