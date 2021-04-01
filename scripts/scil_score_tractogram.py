#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Scores input tractogram overall and bundlewise. Outputs a results.json
containing a full report and splits the input tractogram into resulting
.trk : *_tc.trk, *_fc.trk, nc.trk and *_wpc.trk, where * is the current bundle.

Definitions:
    tc: true connections, streamlines joining a correct combination
        of ROIs.
    fc: false connections, streamlines joining an incorrect combination of
        ROIs.
    nc: no connections, streamlines not joining two ROIs.
    wpc: wrong path connections, streamlines that go outside of the ground
        truth mask, joining a correct combination of ROIs.
    Bundle overlap : ground truth voxels containing tc streamline(s). Either
        input gt_endpoints or gt_heads and gt_tails. Ground truth and ROIs
        must be in the same order i.e. groundTruth1.nii.gz .... \
                groundTruthN.nii.gz --gt_tails tail1.nii.gz ... \
                tailN.nii.gz --gt_heads head1.nii.gz ... headN.nii.gz

Masks can be dilated with --dilate_endpoints for bundle recognition.

"""

import argparse
import json
import itertools
import logging
import nibabel as nib
import numpy as np
import os

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from scipy.ndimage import binary_dilation

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_json_args,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.tractanalysis.reproducibility_measures \
    import (compute_dice_voxel)
# from scilpy.tractanalysis.features import remove_loops_and_sharp_turns
from scilpy.tractanalysis.scoring import (compute_gt_masks,
                                          extract_streamlines,
                                          extract_tails_heads_from_endpoints,
                                          get_binary_maps,
                                          identify_overlapping_roi)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.streamlines import \
    (difference, perform_streamlines_operation)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_tractogram',
                   help="Input tractogram to score(.trk)")
    p.add_argument('gt_bundles', nargs='+',
                   help="Bundles ground truth(.trk, .nii or .nii.gz).")
    g = p.add_argument_group('ROIs')
    g.add_argument('--gt_endpoints', nargs='+',
                   help="Bundles endpoints, both bundle's ROIs\
                       (.nii or .nii.gz).")
    g.add_argument('--gt_tails', nargs='+', required=True,
                   help="Bundles tails, bundle's first ROI(.nii or .nii.gz).")
    g.add_argument('--gt_heads', nargs='+', required=True,
                   help="Bundles heads, bundle's second ROI(.nii or .nii.gz).")
    p.add_argument('--dilate_endpoints', metavar='NB_PASS', default=1, type=int,
                   help='Dilate masks n-times.')
    p.add_argument('--gt_config', metavar='FILE',
                   help=".json dict to specify bundles streamlines min, \
                    max length and max angles.")
    p.add_argument('--out_dir', default='gt_out/',
                   help="Output directory")

    p.add_argument('--wrong_path_as_separate', action='store_true',
                   help="Separates streamlines that go outside of the ground \
                        truth mask from true connections, outputs as \
                        *_wpc.trk.")
    p.add_argument('--remove_duplicate', action='store_true',
                   help='Remove duplicate streamlines before scoring.')
    p.add_argument('--remove_invalid', action='store_true',
                   help='Remove invalid streamlines before scoring.')

    add_json_args(p)
    add_overwrite_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)

    return p


def extract_prefix(mask_1_filename, mask_2_filename):
    prefix_1 = os.path.basename(mask_1_filename)
    prefix_1, _ = split_name_with_nii(prefix_1)
    prefix_2 = os.path.basename(mask_2_filename)
    prefix_2, _ = split_name_with_nii(prefix_2)

    return prefix_1, prefix_2


def find_tc_pos(tc_filenames, filename):
    if filename in tc_filenames:
        return True, tc_filenames.index(filename)
    return False, None


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram] + args.gt_bundles)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)

    if (args.gt_tails and not args.gt_heads) \
            or (args.gt_heads and not args.gt_tails):
        parser.error('Both --gt_heads and --gt_tails are needed.')
    if args.gt_endpoints and (args.gt_tails or args.gt_heads):
        parser.error('Can only provide --gt_endpoints or --gt_tails/gt_heads')
    if not args.gt_endpoints and (not args.gt_tails and not args.gt_heads):
        parser.error(
            'Either input --gt_endpoints or --gt_heads and --gt_tails.')

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    sft = load_tractogram_with_reference(
        parser, args, args.in_tractogram, bbox_check=False)

    logging.info('Computing ground-truth masks')
    gt_bundle_masks, gt_bundle_inv_masks, affine, dimensions,  = \
        compute_gt_masks(args.gt_bundles, parser, args)

    # If endpoints without heads/tails are loaded, split them and continue
    # normally after. Q/C of the output is important
    if args.gt_endpoints:
        logging.info('Extracting ground-truth end and tail masks')
        gt_tails, gt_heads, affine, dimensions = \
            extract_tails_heads_from_endpoints(
                args.gt_endpoints, args.out_dir)
    else:
        gt_tails, gt_heads = args.gt_tails, args.gt_heads

    # Load the endpoints heads/tails, keep the correct combinations
    # separately from all the possible combinations
    tc_filenames = list(zip(gt_tails, gt_heads))

    _, ext = os.path.splitext(args.in_tractogram)

    # if args.gt_config:
    #     with open(args.gt_config, 'r') as json_file:
    #         length_dict = json.load(json_file)

    tc_streamlines_list = []
    wpc_streamlines_list = []

    logging.info('Scoring true connections')
    for (mask_1_filename, mask_2_filename), inv_mask in zip(
        tc_filenames, gt_bundle_inv_masks
    ):
        # Automatically generate filename for Q/C
        prefix_1, prefix_2 = extract_prefix(mask_1_filename, mask_2_filename)
        logging.info('Scoring {} and {}'.format(prefix_1, prefix_2))

        mask_1 = nib.load(mask_1_filename).get_fdata().astype(np.int16)
        mask_2 = nib.load(mask_2_filename).get_fdata().astype(np.int16)

        if args.dilate_endpoints:
            mask_1 = binary_dilation(mask_1, iterations=args.dilate_endpoints)
            mask_2 = binary_dilation(mask_2, iterations=args.dilate_endpoints)

        tmp_sft, sft = extract_streamlines(mask_1, mask_2, sft)

        streamlines = tmp_sft.streamlines
        tc_streamlines = streamlines

        # Config file for each 'bundle'
        # Loops => no connection (nc)
        # Length => false connection (fc)
        # if args.gt_config:
        #     min_len, max_len = \
        #         length_dict[args.gt_bundles[tc_pos]]['length']

        #     lengths = np.array(list(length(streamlines)))
        #     valid_min_length_mask = lengths > min_len
        #     valid_max_length_mask = lengths < max_len
        #     valid_length_mask = np.logical_and(valid_min_length_mask,
        #                                        valid_max_length_mask)
        #     streamlines = ArraySequence(streamlines)

        #     val_len_streamlines = streamlines[valid_length_mask]
        #     fc_streamlines = streamlines[~valid_length_mask]

        #     angle = length_dict[args.gt_bundles[tc_pos]]['angle']
        #     tc_streamlines, loops = remove_loops_and_sharp_turns(
        #         val_len_streamlines, angle)

        #     if loops:
        #         no_conn_streamlines.extend(loops)

        # Streamlines getting out of the bundle mask can be considered
        # separately as wrong path connection (wpc)
        # TODO: Can they ? Seems excessive
        if args.wrong_path_as_separate:
            tmp_sft = StatefulTractogram.from_sft(tc_streamlines, sft)
            wpc_stf, _ = filter_grid_roi(tmp_sft, inv_mask, 'any', False)
            wpc_streamlines = wpc_stf.streamlines
            tc_streamlines, _ = perform_streamlines_operation(
                difference, [tc_streamlines, wpc_streamlines], precision=0)
            wpc_streamlines_list.append(wpc_streamlines)
        else:
            wpc_streamlines = []

        tc_streamlines_list.append(tc_streamlines)
        tc_sft = StatefulTractogram.from_sft(tc_streamlines, sft)
        save_tractogram(tc_sft, os.path.join(
            args.out_dir, '{}_{}_tc{}'.format(prefix_1, prefix_2, ext)),
            bbox_valid_check=False)

        if args.wrong_path_as_separate:
            wpc_sft = StatefulTractogram.from_sft(wpc_streamlines, sft)
            if len(wpc_sft) > 0:
                save_tractogram(wpc_sft,
                                os.path.join(args.out_dir,
                                             '{}_{}_wpc{}'.format(prefix_1,
                                                                  prefix_2,
                                                                  ext)),
                                bbox_valid_check=False)

    overlapping_roi = []
    all_rois = gt_heads + gt_tails
    for roi1, roi2 in itertools.combinations(all_rois, 2):
        overlapping_roi.extend(identify_overlapping_roi(roi1, roi2))

    # if args.gt_config:
    #     with open(args.gt_config, 'r') as json_file:
    #         length_dict = json.load(json_file)

    # Again keep the keep the correct combinations
    comb_filename = list(itertools.combinations(
        itertools.chain(*zip(gt_tails, gt_heads)), r=2))

    # Remove the true connections from all combinations, leaving only
    # false connections
    for tc_f in tc_filenames:
        comb_filename.remove(tc_f)

    fc_streamlines_list = []
    logging.info('Scoring false connections')
    # Go through all the possible combinations of endpoints masks
    for i, roi in enumerate(comb_filename):
        mask_1_filename, mask_2_filename = roi

        # That would be done here.
        # Automatically generate filename for Q/C
        prefix_1, prefix_2 = extract_prefix(mask_1_filename, mask_2_filename)
        _, ext = os.path.splitext(args.in_tractogram)

        logging.info('Scoring {} and {}'.format(prefix_1, prefix_2))

        mask_1 = nib.load(mask_1_filename).get_fdata().astype(np.int16)
        mask_2 = nib.load(mask_2_filename).get_fdata().astype(np.int16)

        if args.dilate_endpoints:
            mask_1 = binary_dilation(mask_1, iterations=args.dilate_endpoints)
            mask_2 = binary_dilation(mask_2, iterations=args.dilate_endpoints)

        tmp_sft, sft = extract_streamlines(mask_1, mask_2, sft)

        streamlines = tmp_sft.streamlines
        fc_streamlines = streamlines

        fc_streamlines_list.append(fc_streamlines)

        fc_sft = StatefulTractogram.from_sft(fc_streamlines, sft)
        if len(fc_sft) > 0:
            save_tractogram(fc_sft, os.path.join(
                args.out_dir, '{}_{}_fc{}'.format(prefix_1,
                                                  prefix_2, ext)),
                            bbox_valid_check=False)

    no_conn_streamlines = sft.streamlines
    final_results = {}
    no_conn_sft = StatefulTractogram.from_sft(no_conn_streamlines, sft)
    save_tractogram(no_conn_sft, os.path.join(
        args.out_dir, 'nc{}'.format(ext)), bbox_valid_check=False)

    # Total number of streamlines for each category
    # and statistic that are not 'bundle-wise'
    tc_streamlines_count = len(list(itertools.chain(*tc_streamlines_list)))
    fc_streamlines_count = len(list(itertools.chain(*fc_streamlines_list)))

    if args.wrong_path_as_separate:
        wpc_streamlines_count = len(
            list(itertools.chain(*wpc_streamlines_list)))
    else:
        wpc_streamlines_count = 0

    nc_streamlines_count = len(no_conn_streamlines)
    total_count = tc_streamlines_count + fc_streamlines_count + \
        wpc_streamlines_count + nc_streamlines_count

    final_results['tractogram_filename'] = str(args.in_tractogram)
    final_results['tractogram_overlap'] = 0.0
    final_results['tc_streamlines'] = tc_streamlines_count
    final_results['fc_streamlines'] = fc_streamlines_count
    final_results['nc_streamlines'] = nc_streamlines_count

    final_results['tc_bundle'] = len([x for x in tc_streamlines_list if x])
    final_results['fc_bundle'] = len([x for x in fc_streamlines_list if x])

    final_results['tc_streamlines_ratio'] = tc_streamlines_count / total_count
    final_results['fc_streamlines_ratio'] = fc_streamlines_count / total_count
    final_results['nc_streamlines_ratio'] = nc_streamlines_count / total_count

    if args.wrong_path_as_separate:
        final_results['wpc_streamlines'] = wpc_streamlines_count
        final_results['wpc_streamlines_ratio'] = \
            wpc_streamlines_count / total_count
        final_results['wpc_bundle'] = len(
            [x for x in wpc_streamlines_list if x])

    final_results['total_streamlines'] = total_count
    final_results["bundle_wise"] = {}
    final_results["bundle_wise"]["true_connections"] = {}
    final_results["bundle_wise"]["false_connections"] = {}
    tractogram_overlap = 0.0

    for i, filename in enumerate(tc_filenames):
        current_tc_streamlines = tc_streamlines_list[i]
        current_tc_voxels, current_tc_endpoints_voxels = get_binary_maps(
            current_tc_streamlines, dimensions, sft, args.remove_invalid)

        if args.wrong_path_as_separate:
            current_wpc_streamlines = wpc_streamlines_list[i]
            current_wpc_voxels, _ = get_binary_maps(
                current_wpc_streamlines, dimensions, sft, args.remove_invalid)

        tmp_dict = {}
        tmp_dict['tc_streamlines'] = len(current_tc_streamlines)

        tmp_dict['tc_dice'] = compute_dice_voxel(gt_bundle_masks[i],
                                                 current_tc_voxels)[0]

        bundle_overlap = gt_bundle_masks[i] * current_tc_voxels
        bundle_overreach = np.zeros(dimensions)
        bundle_overreach[np.where(
            (gt_bundle_masks[i] == 0) & (current_tc_voxels >= 1))] = 1
        bundle_lacking = np.zeros(dimensions)
        bundle_lacking[np.where(
            (gt_bundle_masks[i] == 1) & (current_tc_voxels == 0))] = 1

        tmp_dict['tc_bundle_overlap'] = np.count_nonzero(bundle_overlap)
        tmp_dict['tc_bundle_overreach'] = \
            np.count_nonzero(bundle_overreach)
        tmp_dict['tc_bundle_lacking'] = np.count_nonzero(bundle_lacking)
        tmp_dict['tc_bundle_overlap_PCT'] = \
            tmp_dict['tc_bundle_overlap'] / \
            (tmp_dict['tc_bundle_overlap'] +
                tmp_dict['tc_bundle_lacking'])
        tractogram_overlap += tmp_dict['tc_bundle_overlap_PCT']

        endpoints_overlap = \
            gt_bundle_masks[i] * current_tc_endpoints_voxels
        endpoints_overreach = np.zeros(dimensions)
        endpoints_overreach[np.where(
            (gt_bundle_masks[i] == 0) &
            (current_tc_endpoints_voxels >= 1))] = 1
        tmp_dict['tc_endpoints_overlap'] = np.count_nonzero(
            endpoints_overlap)
        tmp_dict['tc_endpoints_overreach'] = np.count_nonzero(
            endpoints_overreach)

        if args.wrong_path_as_separate:
            tmp_dict['wpc_streamlines'] = len(current_wpc_streamlines)
            tmp_dict['wpc_dice'] = \
                compute_dice_voxel(gt_bundle_masks[i],
                                   current_wpc_voxels)[0]

        final_results["bundle_wise"]["true_connections"][str(filename)] = \
            tmp_dict

    # Bundle-wise statistics, useful for more complex phantom
    for i, filename in enumerate(comb_filename):
        is_tp, tc_pos = find_tc_pos(tc_filenames, filename)
        current_fc_streamlines = fc_streamlines_list[i]
        current_fc_voxels, _ = get_binary_maps(
            current_fc_streamlines, dimensions, sft, args.remove_invalid)

        tmp_dict = {}

        if len(current_fc_streamlines):
            tmp_dict['fc_streamlines'] = len(current_fc_streamlines)
            tmp_dict['fc_voxels'] = np.count_nonzero(current_fc_voxels)

            final_results["bundle_wise"]["false_connections"][str(filename)] =\
                tmp_dict

    final_results['tractogram_overlap'] = \
        tractogram_overlap / len(gt_bundle_masks)

    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(final_results, f,
                  indent=args.indent,
                  sort_keys=args.sort_keys)


if __name__ == '__main__':
    main()
