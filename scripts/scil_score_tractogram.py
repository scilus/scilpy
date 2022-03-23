#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Scores input tractogram overall and bundlewise. Outputs a results.json
containing a full report and splits the input tractogram into resulting
tractogram : *_tc.tck, *_fc.tck, nc.tck and *_wpc.tck,
where * is the current bundle.

Definitions:
    - tc: true connections, streamlines joining a correct combination
        of ROIs.
    - fc: false connections, streamlines joining an incorrect combination of
        ROIs.
    - nc: no connections, streamlines not joining two ROIs.
    - wpc: wrong path connections, streamlines that go outside of the ground
        truth mask, joining a correct combination of ROIs.
    - Bundle overlap : ground truth voxels containing tc streamline(s).

Masks can be dilated with --dilate_endpoints for bundle recognition.

Config dictionnary needs to be a json containing a dict of the ground-truth
bundles as keys and the value being a dictionnary with
    - endpoints OR head/tail: filename for the endpoints ROI.
        If 'enpoints' is used, we will automatically separate the mask into
        two ROIs, acting as head and tail. QC is strongly recommended.
    - bundle_mask (optional): if set, streamlines outside this ground truth
        path will be defined as wpc. Files must be .tck, .trk, .nii or .nii.gz
        filenames. If it is is a tractogram, a mask will be created. If it is a
        nifti file, it will be considered as a mask.
    - angle (optional): if set, we will remove loops and sharp turns (up to
        given angle) for the bundle.**
    - length (optional): maximum and minimum lengths per bundle. Streamlines
        outside this range will be classified as false connections even if they
        do connect the right ROIs.**

**Rejected streamlines will be classified as wrong path connections.

Ex 1:
{
  "Ground_truth_bundle_0": {
    "angle": 300,
    "length": [140, 150],
    "endpoints": PATH/'file1'
  }
}

Ex 2:
{
  "Ground_truth_bundle_1": {
    "head": 'file2',
    "tail": 'file3',
    "bundle_mask": ground_truth_bundle_1.nii.gz
  }
}
(used with options --bundle_masks_dir PATH1 --rois_dir PATH2)

"""

import argparse
import json
import itertools
import logging
import numpy as np
import os

from dipy.io.utils import is_header_compatible
from dipy.io.stateful_tractogram import StatefulTractogram, \
    set_sft_logger_level
from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_json_args,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.reproducibility_measures \
    import (compute_dice_voxel)
from scilpy.tractanalysis.scoring import (compute_gt_masks,
                                          extract_false_connections,
                                          extract_true_connections,
                                          get_binary_maps,
                                          compute_endpoint_masks,
                                          make_sft_from_ids)
from scilpy.utils.filenames import split_name_with_nii


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("in_tractogram",
                   help="Input tractogram to score")
    p.add_argument("gt_config",
                   help=".json dict to specify bundles streamlines min, \n"
                        "max length and max angles.")
    p.add_argument("out_dir",
                   help="Output directory")

    p.add_argument("--bundle_masks_dir",
                   help="Path of the bundle paths listed in the gt_config.\n "
                        "If not set, filenames in the config file are "
                        "considered as complete paths.")
    p.add_argument("--rois_dir",
                   help="Path of the ROI files listed in the gt_config.\n If "
                        "not set, filenames in the config file are considered "
                        "as complete paths.")
    p.add_argument("--dilate_endpoints",
                   metavar="NB_PASS", default=1, type=int,
                   help="Dilate masks n-times.")
    p.add_argument("--remove_invalid", action="store_true",
                   help="Remove invalid streamlines before scoring.")
    p.add_argument("--no_empty", action='store_true',
                   help='Do not write file if there is no streamline.')

    add_json_args(p)
    add_overwrite_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)

    return p


def extract_prefix(filename):
    prefix = os.path.basename(filename)
    prefix, _ = split_name_with_nii(prefix)

    return prefix


def read_config_file(gt_config, bundle_masks_dir, rois_dir):
    # Create
    # roi_options = {
    #      'bundle1': {
    #              'gt_endpoints': path + file,  # OR
    #              'gt_head': path + file
    #              'gt_tail': path + file}}
    angles = []
    lengths = []
    masks = []
    roi_options = {}
    roi_files = []

    with open(gt_config, "r") as json_file:
        config = json.load(json_file)

        bundles = list(config.keys())
        for bundle in bundles:
            bundle_config = config[bundle]
            if 'angle' in bundle_config:
                angles.append(bundle_config['angle'])
            else:
                angles.append(None)

            if 'length' in bundle_config:
                lengths.append(bundle_config['length'])
            else:
                lengths.append(None)

            if 'bundle_mask' in bundle_config:
                masks.append(os.path.join(bundle_masks_dir,
                                          bundle_config['bundle_mask']))
            else:
                masks.append(None)

            if 'endpoints' in bundle_config:
                if 'head' in bundle_config or 'tail' in bundle_config:
                    raise ValueError("Bundle {} has confusion keywords in the "
                                     "config file. Please choose either "
                                     "endpoints OR head/tail.".format(bundle))
                endpoints = os.path.join(rois_dir, bundle_config['endpoints'])
                roi_options.update({
                    bundle: {'endpoints': endpoints}})
                roi_files.append(endpoints)
            elif 'head' in bundle_config:
                if 'tail' not in bundle_config:
                    raise ValueError("You have provided the head for bundle "
                                     "{}, but not the tail".format(bundle))
                head = os.path.join(rois_dir, bundle_config['head'])
                tail = os.path.join(rois_dir, bundle_config['tail'])
                roi_options.update({
                    bundle: {
                        'gt_head': head,
                        'gt_tail': tail
                    }})
                roi_files.append(head)
                roi_files.append(tail)
            else:
                raise ValueError("Bundle configuration for bundle {} misses "
                                 "'endpoints' or 'head'/'tail'".format(bundle))

    return bundles, masks, roi_options, roi_files, lengths, angles


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.gt_config)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)

    # -----------
    # Preparation
    # -----------
    # Read the config file
    bundles_names, masks, roi_options, all_rois, lengths, angles = \
        read_config_file(args.gt_config, args.bundle_masks_dir, args.rois_dir)
    nb_bundles = len(bundles_names)

    # Remove duplicates
    # (in case a same roi file is used for more than one bundle)
    all_rois = list(dict.fromkeys(all_rois))

    # Verify options
    assert_inputs_exist(parser, masks + all_rois)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Loading tractogram.")
    _, ext = os.path.splitext(args.in_tractogram)
    sft = load_tractogram_with_reference(
        parser, args, args.in_tractogram, bbox_check=False)

    if args.remove_invalid:
        sft.remove_invalid_streamlines()

    initial_count = len(sft)

    logging.info("Verifying compatibility of tractogram with bundle masks.")
    for gt in masks:
        _, gt_ext = os.path.splitext(gt)
        if gt_ext in ['.trk', '.tck']:
            gt_bundle = load_tractogram_with_reference(
                parser, args, gt, bbox_check=False)
        else:
            gt_bundle = gt
        compatible = is_header_compatible(sft, gt_bundle)
        if not compatible:
            parser.error("Input tractogram incompatible with"
                         " {}".format(gt))

    logging.info("Loading and/or computing ground-truth masks.")
    gt_bundle_masks, gt_bundle_inv_masks, affine, dimensions, = \
        compute_gt_masks(masks, parser, args)

    logging.info("Extracting ground-truth head and tail masks.")
    gt_tails, gt_heads = compute_endpoint_masks(
        roi_options, affine, dimensions, args.out_dir)

    logging.info("Verifying tractogram compatibility with endpoint ROIs.")
    for gt in gt_tails + gt_heads:
        compatible = is_header_compatible(sft, gt)
        if not compatible:
            parser.error("Input tractogram incompatible with {}".format(gt))

    # -----------
    # True connections
    # -----------
    logging.info("Scoring true connections (and wpc)")

    # List the heads/tails combinations
    tc_filenames = list(zip(gt_tails, gt_heads))

    tc_sft_list = []
    tc_ids_list = []
    wpc_sft_list = []
    wpc_ids_list = []
    for i, (mask_1_filename, mask_2_filename) in enumerate(tc_filenames):

        # Automatically generate filename for Q/C
        prefix_1 = extract_prefix(mask_1_filename)
        prefix_2 = extract_prefix(mask_2_filename)

        # Extract true connection
        tc_sft, wpc_sft, tc_ids, wpc_ids = extract_true_connections(
            sft, mask_1_filename, mask_2_filename, lengths[i], angles[i],
            bundles_names[i], gt_bundle_inv_masks[i],
            args.dilate_endpoints)

        # Save results
        if len(tc_sft) > 0 or not args.no_empty:
            save_tractogram(tc_sft, os.path.join(
                args.out_dir, "{}_{}_tc{}".format(prefix_1, prefix_2, ext)),
                            bbox_valid_check=False)

        if len(wpc_sft) > 0 or not args.no_empty:
            save_tractogram(wpc_sft, os.path.join(
                args.out_dir, "{}_{}_wpc{}".format(prefix_1, prefix_2, ext)),
                            bbox_valid_check=False)

        tc_sft_list.append(tc_sft)
        tc_ids_list.append(tc_ids)
        wpc_sft_list.append(wpc_sft)
        wpc_ids_list.append(wpc_ids)

    # Duplicates?
    tc_and_wpc = [tc_ids_list[i] + wpc_ids_list[i] for i in
                  range(len(tc_ids_list))]
    for i in range(nb_bundles):
        for j in range(i + 1, nb_bundles):
            duplicate_ids = np.intersect1d(tc_and_wpc[i], tc_and_wpc[j])
            if len(duplicate_ids) > 0:
                logging.warning(
                    "{} streamlines belong both to bundle {} and {}. \n"
                    "Please verify your criteria!"
                    .format(len(duplicate_ids), bundles_names[i],
                            bundles_names[j]))

    # -----------
    # False connections
    # -----------
    logging.info("Scoring false connections")

    # Keep all possible combinations
    comb_filename = list(itertools.combinations(all_rois, r=2))

    # Remove the true connections from all combinations, leaving only
    # false connections
    for tc_f in tc_filenames:
        tc_f = tuple(sorted(tc_f))
        comb_filename.remove(tc_f)

    # Go through all the possible combinations of endpoints masks
    fc_sft_list = []
    fc_ids_list = []
    for i, roi in enumerate(comb_filename):
        mask_1_filename, mask_2_filename = roi

        # Automatically generate filename for Q/C
        prefix_1 = extract_prefix(mask_1_filename)
        prefix_2 = extract_prefix(mask_2_filename)
        _, ext = os.path.splitext(args.in_tractogram)

        fc_sft, fc_ids = extract_false_connections(
            sft, mask_1_filename, mask_2_filename, args.dilate_endpoints)

        if len(fc_sft) > 0 or not args.no_empty:
            save_tractogram(fc_sft, os.path.join(
                args.out_dir, "{}_{}_fc{}".format(prefix_1, prefix_2, ext)),
                            bbox_valid_check=False)

        logging.info("Recognized {} streamlines between {} and {}".format(
            len(fc_sft.streamlines), prefix_1, prefix_2))

        fc_sft_list.append(fc_sft)
        fc_ids_list.append(fc_ids)

    # Duplicates?
    nb_pairs = len(fc_ids_list)
    for i in range(nb_pairs):
        for j in range(i + 1, nb_pairs):
            duplicate_ids = np.intersect1d(fc_ids_list[i], fc_ids_list[j])
            if len(duplicate_ids) > 0:
                logging.warning(
                    "{} streamlines are scored twice as invalid connections \n"
                    "(between pair {}\n and between pair {}).\n You probably "
                    "have overlapping ROIs!"
                    .format(len(duplicate_ids), comb_filename[i],
                            comb_filename[j]))

    # -----------
    # No connections
    # -----------
    # No connections = ids that are not tc, not wpc and not fc.
    all_tc_ids = np.unique(list(itertools.chain(*tc_ids_list)))
    all_wpc_ids = np.unique(list(itertools.chain(*wpc_ids_list)))
    all_fc_ids = np.unique(list(itertools.chain(*fc_ids_list)))
    remaining_ids = np.arange(len(sft))
    remaining_ids = np.setdiff1d(remaining_ids, all_tc_ids)
    remaining_ids = np.setdiff1d(remaining_ids, all_wpc_ids)
    remaining_ids = np.setdiff1d(remaining_ids, all_fc_ids)

    logging.info("The remaining {} / {} streamlines will be scored as nc."
                 .format(len(remaining_ids), len(sft)))

    no_conn_sft = make_sft_from_ids(remaining_ids, sft)
    if len(no_conn_sft) > 0 or not args.no_empty:
        save_tractogram(no_conn_sft, os.path.join(
            args.out_dir, "nc{}".format(ext)), bbox_valid_check=False)

    # -----------
    # Tractometry stats: NC, IC, VC, WPC
    # -----------
    # Total number of streamlines for each category
    # and statistic that are not "bundle-wise"
    tc_streamlines_count = len(all_tc_ids)
    wpc_streamlines_count = len(all_wpc_ids)
    fc_streamlines_count = len(all_fc_ids)
    nc_streamlines_count = len(remaining_ids)

    total_count = tc_streamlines_count + fc_streamlines_count + \
        wpc_streamlines_count + nc_streamlines_count

    assert total_count == initial_count

    final_results = {
        "tractogram_filename": str(args.in_tractogram),
        "bundles": bundles_names,
        "tractogram_overlap": 0.0,
        "tc_streamlines": tc_streamlines_count,
        "wpc_streamlines": wpc_streamlines_count,
        "fc_streamlines": fc_streamlines_count,
        "nc_streamlines": nc_streamlines_count,
        "tc_bundle": len([x for x in tc_ids_list if len(x) > 0]),
        "fc_bundle": len([x for x in fc_ids_list if len(x) > 0]),
        "wpc_bundle": len([x for x in wpc_ids_list if len(x) > 0]),
        "tc_streamlines_ratio": tc_streamlines_count / total_count,
        "fc_streamlines_ratio": fc_streamlines_count / total_count,
        "nc_streamlines_ratio": nc_streamlines_count / total_count,
        "wpc_streamlines_ratio": wpc_streamlines_count / total_count,
        "total_streamlines": total_count,
    }

    # -----------
    # Tractometry stats: OL, OR, Dice score
    # -----------
    tractogram_overlap = 0.0

    tc_bundle_wise_dict = {}
    for i, filename in enumerate(tc_filenames):
        current_tc_streamlines = tc_sft_list[i].streamlines
        current_wpc_streamlines = wpc_sft_list[i].streamlines

        # Getting the recovered mask
        current_tc_voxels, current_tc_endpoints_voxels = get_binary_maps(
            current_tc_streamlines, sft)
        current_wpc_voxels, _ = get_binary_maps(current_wpc_streamlines, sft)

        # Dice
        tc_dice = compute_dice_voxel(gt_bundle_masks[i], current_tc_voxels)
        wpc_dice = compute_dice_voxel(gt_bundle_masks[i], current_wpc_voxels)

        # Overlap and overreach
        bundle_overlap = gt_bundle_masks[i] * current_tc_voxels
        bundle_overreach = np.zeros(dimensions)
        # If no ground truth bundle was given (only the endpoints ROIs),
        # overreach can be computed as usual
        bundle_overreach[np.where(
            (gt_bundle_masks[i] == 0) & (current_tc_voxels >= 1))] = 1
        # If a ground truth bundle has been given, all streamlines contributing
        # to the overreach are now classified as wpc.
        bundle_overreach[np.where(
            (gt_bundle_masks[i] == 0) & (current_wpc_voxels >= 1))] = 1

        bundle_lacking = np.zeros(dimensions)
        bundle_lacking[np.where(
            (gt_bundle_masks[i] == 1) & (current_tc_voxels == 0))] = 1

        overlap = np.count_nonzero(bundle_overlap)
        overreach = np.count_nonzero(bundle_overreach)
        lacking = np.count_nonzero(bundle_lacking)

        # Endpoints coverage
        endpoints_overlap = gt_bundle_masks[i] * current_tc_endpoints_voxels
        endpoints_overreach = np.zeros(dimensions)
        endpoints_overreach[np.where(
            (gt_bundle_masks[i] == 0) &
            (current_tc_endpoints_voxels >= 1))] = 1

        tmp_dict = {
            "bundle": bundles_names[i],
            "tc_streamlines": len(current_tc_streamlines),
            "wpc_streamlines": len(current_wpc_streamlines),
            "tc_dice": tc_dice[0],
            "wpc_dice": wpc_dice[0],
            "tc_bundle_overlap": overlap,
            "tc_bundle_overreach": overreach,
            "tc_bundle_lacking": lacking,
            "tc_bundle_overlap_PCT": overlap / (overlap + lacking),
            "tc_endpoints_overlap": np.count_nonzero(
                endpoints_overlap),
            "tc_endpoints_overreach": np.count_nonzero(
                endpoints_overreach)}

        tractogram_overlap += tmp_dict["tc_bundle_overlap_PCT"]
        tc_bundle_wise_dict.update({str(filename): tmp_dict})

    # -----------
    # False connections stats: number of voxels
    # -----------
    fc_bundle_wise_dict = {}
    for i, filename in enumerate(comb_filename):
        current_fc_streamlines = fc_sft_list[i].streamlines

        if len(current_fc_streamlines):
            current_fc_voxels, _ = get_binary_maps(
                current_fc_streamlines, sft)

            tmp_dict = {
                "fc_streamlines": len(current_fc_streamlines),
                "fc_voxels": np.count_nonzero(current_fc_voxels)
            }
            fc_bundle_wise_dict.update({str(filename): tmp_dict})

    bundle_wise_dict = {
        "true_connections": tc_bundle_wise_dict,
        "false_connections": fc_bundle_wise_dict
    }

    final_results.update({
        "bundle_wise": bundle_wise_dict,
        "tractogram_overlap": tractogram_overlap / len(gt_bundle_masks)
    })

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(final_results, f,
                  indent=args.indent,
                  sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
