#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Scores input tractogram overall and bundlewise. Outputs a results.json
containing a full report and splits the input tractogram into resulting
tractogram : *_tc.tck, *_fc.tck, nc.tck and *_wpc.tck,
where * is the current bundle.

Definitions:
    In terms of number of streamlines:
    - tc: true connections, streamlines joining a correct combination
        of ROIs.
    - fc: false connections, streamlines joining an incorrect combination of
        ROIs.
    - nc: no connections, streamlines not joining two ROIs.
    - wpc: wrong path connections, streamlines that go outside of the ground
        truth mask, joining a correct combination of ROIs. One wpc file per
        bundle will be saved. They could contain duplicated if your ROIs
        overlap. The final total wpc file, however, will not contain
        duplicates.

    In terms of number of voxels:
    - Bundle overlap : ground truth voxels containing tc streamline(s).

Config dictionnary needs to be a json containing a dict of the ground-truth
bundles as keys and the value being a dictionnary with
    - endpoints OR head/tail: filename for the endpoints ROI.
        If 'enpoints' is used, we will automatically separate the mask into
        two ROIs, acting as head and tail. QC is strongly recommended.
    - gt_mask: Expected result. Overreach and overlap metrics (OR, OL) will be
        computed from this mask.*
    - limits_mask (optional): if set, streamlines outside this path will be
        defined as wrong path connections (wpc) if they are not included in
        any other bundle as tc. This is thus the equivalent of 'include' 'all'
        in our typical filtering scripts.*
    - angle (optional): if set, we will remove loops and sharp turns (up to
        given angle) for the bundle. Removed streamlines will be classified as
        wpc.**
    - length (optional): maximum and minimum lengths per bundle. Streamlines
        outside this range will be classified as wpc.**

* Files must be .tck, .trk, .nii or .nii.gz. If it is is a tractogram, a mask
will be created. If it is a nifti file, it will be considered as a mask.
** Rejected streamlines will be classified as wrong path connections.

Ex 1:
{
  "Ground_truth_bundle_0": {
    "gt_mask": "PATH/bundle0.nii.gz",
    "angle": 300,
    "length": [140, 150],
    "endpoints": PATH/'file1'
  }
}

Ex 2:
{
  "Ground_truth_bundle_1": {
    "gt_mask": "bundle1.trk"
    "head": 'file2',
    "tail": 'file3',
    "limits_mask": big_bundle_1.nii.gz
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
from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_json_args,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel
from scilpy.tractanalysis.scoring import (compute_masks,
                                          extract_false_connections,
                                          get_binary_maps,
                                          compute_endpoint_masks,
                                          make_sft_from_ids,
                                          extract_true_connections)
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

    g = p.add_argument_group("Additions to gt_config")
    g.add_argument("--gt_masks_dir", default='',
                   help="Path of the gt_masks listed in the gt_config.\n "
                        "If not set, filenames in the config file are "
                        "considered as complete paths.")
    g.add_argument("--limits_masks_dir", default='',
                   help="Path of the limits_masks listed in the "
                        "gt_config.\n If not set, filenames in the config "
                        "file are considered as complete paths.")
    g.add_argument("--rois_dir", default='',
                   help="Path of the ROI files listed in the gt_config (head, "
                        "tail of endpoints).\n If not set, filenames in the "
                        "config file are considered as complete paths.")
    g.add_argument("--use_gt_masks_as_limits_masks", action='store_true',
                   help="If set, the gt_config's 'gt_mask' will also be used "
                        "as 'limits_mask' for each bundle. Note that this "
                        "means the OR will necessarily be 0.")

    g = p.add_argument_group("Preprocessing")
    g.add_argument("--dilate_endpoints",
                   metavar="NB_PASS", default=0, type=int,
                   help="Dilate inclusion masks n-times. Default: 0.")
    g.add_argument("--remove_invalid", action="store_true",
                   help="Remove invalid streamlines before scoring.")

    g = p.add_argument_group("Tractometry choices")
    g.add_argument("--compute_fc", action='store_true',
                   help="If set, false connections will be separated in sub-"
                        "bundles, one for each pair of ROI not belonging to "
                        "a true connection. Else, all streamlines not "
                        "included as tc or wpc will be classified as nc.")
    g.add_argument("--remove_wpc_belonging_to_other_gt", action='store_true',
                   help="If set, wpc belonging to another gt bundle (in the "
                        "case of overlapping ROIs) will be removed from the "
                        "wpc classification.")

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


def _verify_compatibility_with_bundles(sft, masks_files, parser, args):
    """
    Verifies the compatibility of the main sft with the bundle masks, which can
    be either tractograms or nifti files.
    """
    for file in masks_files:
        _, ext = os.path.splitext(file)
        if ext in ['.trk', '.tck']:
            mask = load_tractogram_with_reference(parser, args, file,
                                                  bbox_check=False)
        else:
            mask = file
        compatible = is_header_compatible(sft, mask)
        if not compatible:
            parser.error("Input tractogram incompatible with {}".format(file))


def read_config_file(args):
    """
    Read the gt_config file and returns:

    angles: the list of maximum angles per bundle (None if not set)
    lengths: the list of [min max] lengths per bundle (None if not set)
    gt_masks: the list of gt_mask filenames per bundle (None if not set)
    limits_masks: the list of limits_masks filenames per bundles (None if not
          set)
    roi_options: a dict with, for each bundle, the keys 'gt_head', 'gt_tail' if
          they are set, else the key 'gt_endpoints'.
    """
    angles = []
    lengths = []
    gt_masks = []
    limits_masks = []
    roi_options = {}

    with open(args.gt_config, "r") as json_file:
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

            if 'gt_mask' in bundle_config:
                gt_masks.append(os.path.join(args.gt_masks_dir,
                                             bundle_config['gt_mask']))
            else:
                logging.warning(
                    "No gt_mask set for bundle {}. Some tractometry metrics "
                    "won't be computed (OR, OL).".format(bundle))
                gt_masks.append(None)

            if 'limits_mask' in bundle_config:
                if args.use_gt_masks_as_limits_masks:
                    raise ValueError(
                        "With the option --use_gt_masks_as_limits_masks, "
                        "you should not add any limits_mask in the config "
                        "file.")
                limits_masks.append(os.path.join(args.limits_masks_dir,
                                                 bundle_config['limits_mask']))
            else:
                if args.use_gt_masks_as_limits_masks:
                    limits_masks.append(gt_masks[-1])
                else:
                    limits_masks.append(None)

            if 'endpoints' in bundle_config:
                if 'head' in bundle_config or 'tail' in bundle_config:
                    raise ValueError(
                        "Bundle {} has confusion keywords in the config file. "
                        "Please choose either endpoints OR head/tail."
                        .format(bundle))
                endpoints = os.path.join(args.rois_dir,
                                         bundle_config['endpoints'])
                roi_options.update({bundle: {'gt_endpoints': endpoints}})
            elif 'head' in bundle_config:
                if 'tail' not in bundle_config:
                    raise ValueError(
                        "You have provided the head for bundle {}, but not "
                        "the tail".format(bundle))
                head = os.path.join(args.rois_dir, bundle_config['head'])
                tail = os.path.join(args.rois_dir, bundle_config['tail'])
                roi_options.update({bundle: {'gt_head': head,
                                             'gt_tail': tail
                                             }})
            else:
                raise ValueError(
                    "Bundle configuration for bundle {} misses 'endpoints' or "
                    "'head'/'tail'".format(bundle))

    return bundles, gt_masks, limits_masks, roi_options, lengths, angles


def compute_true_connections_all_bundles(
        gt_tails, gt_heads, sft, lengths, angles, bundles_names,
        limits_inv_masks, args, ext, path_duplicates):
    """
    Loop on all bundles and extract true connections and wpc.

    True connections:
       1) Connect the head and tail
       2) Are completely included in the limits_mask (if any)
       3) Have acceptable angle and length.
     +
    WPC connections:
       1) connect the head and tail but criteria 2 and 3 are not respected
    """
    nb_bundles = len(bundles_names)

    tc_sft_list = []
    tc_ids_list = []
    wpc_ids_list = []
    bundles_stats = []
    for i in range(nb_bundles):
        head_filename = gt_heads[i]
        tail_filename = gt_tails[i]

        # Extract true connection
        tc_ids, wpc_ids, bundle_stats = \
            extract_true_connections(sft, head_filename, tail_filename,
                                     lengths[i], angles[i], bundles_names[i],
                                     limits_inv_masks[i],
                                     args.dilate_endpoints)

        tc_sft = make_sft_from_ids(tc_ids, sft)

        # Save results
        if len(tc_sft) > 0 or not args.no_empty:
            save_tractogram(tc_sft, os.path.join(
                args.out_dir,
                "segmented_VB/{}_tc{}".format(bundles_names[i], ext)),
                            bbox_valid_check=False)

        tc_sft_list.append(tc_sft)
        tc_ids_list.append(tc_ids)
        wpc_ids_list.append(wpc_ids)
        bundles_stats.append(bundle_stats)

        logging.info("Bundle {}: nb tc = {}"
                     .format(bundles_names[i], bundle_stats["TC"]))

    # Duplicates?
    for i in range(nb_bundles):
        for j in range(i + 1, nb_bundles):
            duplicate_ids = np.intersect1d(tc_ids_list[i], tc_ids_list[j])
            if len(duplicate_ids) > 0:
                logging.warning(
                    "{} streamlines belong both to true connections of both "
                    "bundles {} and {}. Please verify your criteria!"
                    .format(len(duplicate_ids), bundles_names[i],
                            bundles_names[j]))
                if not os.path.isdir(path_duplicates):
                    os.makedirs(path_duplicates)
                save_tractogram(sft[duplicate_ids], os.path.join(
                    path_duplicates, 'duplicates_' + bundles_names[i] + '_' +
                    bundles_names[j] + '.trk'))

    return tc_sft_list, tc_ids_list, wpc_ids_list, bundles_stats


def clean_wpc_all_bundles(wpc_ids_list, sft, bundles_names, args, ext,
                          tc_ids_list, bundles_stats):
    """
    Clean wpc: for now, just saves the wpc per bundle.
    """
    nb_bundles = len(wpc_ids_list)
    wpc_sft_list = []
    for i in range(nb_bundles):
        wpc_ids = wpc_ids_list[i]

        if args.remove_wpc_belonging_to_other_gt:
            all_other_gt = list(itertools.chain(
                *[tc_ids_list[j] for j in range(nb_bundles) if j != i]))
            wpc_ids = np.setdiff1d(wpc_ids, all_other_gt)

        wpc_sft = make_sft_from_ids(wpc_ids, sft)
        wpc_sft_list.append(wpc_sft)

        if len(wpc_ids) > 0 or not args.no_empty:
            save_tractogram(wpc_sft, os.path.join(
                args.out_dir,
                "segmented_VB/{}_wpc{}".format(bundles_names[i], ext)),
                            bbox_valid_check=False)
        bundles_stats[i].update({"Cleaned wpc": len(wpc_ids)})

        logging.info("Bundle {}: nb wpc = {}"
                     .format(bundles_names[i], len(wpc_ids)))

    return wpc_sft_list, bundles_stats


def compute_false_connections_all_bundles(comb_filename, sft, args):
    """
    Loop on all bundles and compute false connections, defined as connections
    between ROIs pairs that do not form gt bundles.

    (Goes through all the possible combinations of endpoints masks)
    """
    fc_sft_list = []
    fc_ids_list = []
    for i, roi in enumerate(comb_filename):
        roi1_filename, roi2_filename = roi

        # Automatically generate filename for Q/C
        prefix_1 = extract_prefix(roi1_filename)
        prefix_2 = extract_prefix(roi2_filename)
        _, ext = os.path.splitext(args.in_tractogram)

        fc_sft, fc_ids = extract_false_connections(
            sft, roi1_filename, roi2_filename, args.dilate_endpoints)

        if len(fc_sft) > 0 or not args.no_empty:
            save_tractogram(fc_sft, os.path.join(
                args.out_dir,
                "segmented_IB/{}_{}_fc{}".format(prefix_1, prefix_2, ext)),
                            bbox_valid_check=False)

        if len(fc_sft.streamlines) > 0:
            logging.info("Recognized {} streamlines between {} and {}"
                         .format(len(fc_sft.streamlines), prefix_1, prefix_2))

        fc_sft_list.append(fc_sft)
        fc_ids_list.append(fc_ids)

    # Duplicates?
    nb_pairs = len(fc_ids_list)
    for i in range(nb_pairs):
        for j in range(i + 1, nb_pairs):
            duplicate_ids = np.intersect1d(fc_ids_list[i], fc_ids_list[j])
            if len(duplicate_ids) > 0:
                logging.warning(
                    "{} streamlines are scored twice as invalid "
                    "connections \n (between pair {}\n and between pair "
                    "{}).\n You probably have overlapping ROIs!"
                    .format(len(duplicate_ids), comb_filename[i],
                            comb_filename[j]))

    return fc_sft_list, fc_ids_list


def compute_tractometry(all_tc_ids, all_wpc_ids, all_fc_ids, all_nc_ids,
                        tc_ids_list, wpc_ids_list, fc_ids_list,
                        tc_sft_list, wpc_sft_list, fc_sft_list, sft,
                        args, bundles_names, gt_masks, dimensions,
                        comb_filename):
    """
    Tractometry stats: First in terms of connections (NC, IC, VC, WPC), then
    in terms of volume (OL, OR, Dice score)
    """
    nb_bundles = len(bundles_names)

    # Total number of streamlines for each category
    tc_streamlines_count = len(all_tc_ids)
    wpc_streamlines_count = len(all_wpc_ids)
    fc_streamlines_count = len(all_fc_ids)
    nc_streamlines_count = len(all_nc_ids)
    total_count = len(sft)

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

    # Tractometry stats over volume: OL, OR, Dice score
    tractogram_overlap = 0.0

    tc_bundle_wise_dict = {}
    for i in range(nb_bundles):
        current_tc_streamlines = tc_sft_list[i].streamlines
        current_wpc_streamlines = wpc_sft_list[i].streamlines

        # Getting the recovered mask
        current_tc_voxels, current_tc_endpoints_voxels = get_binary_maps(
            current_tc_streamlines, sft)
        current_wpc_voxels, _ = get_binary_maps(current_wpc_streamlines, sft)

        if gt_masks[i] is not None:
            # Dice
            tc_dice = compute_dice_voxel(gt_masks[i], current_tc_voxels)[0]
            wpc_dice = compute_dice_voxel(gt_masks[i], current_wpc_voxels)[0]

            # Overlap and overreach
            bundle_overlap = gt_masks[i] * current_tc_voxels
            bundle_overreach = np.zeros(dimensions)
            # If no ground truth bundle was given (only the endpoints ROIs),
            # overreach can be computed as usual
            bundle_overreach[np.where(
                (gt_masks[i] == 0) & (current_tc_voxels >= 1))] = 1
            # If a ground truth bundle has been given, all streamlines
            # contributing to the overreach are now classified as wpc.
            bundle_overreach[np.where(
                (gt_masks[i] == 0) & (current_wpc_voxels >= 1))] = 1

            bundle_lacking = np.zeros(dimensions)
            bundle_lacking[np.where(
                (gt_masks[i] == 1) & (current_tc_voxels == 0))] = 1

            overlap = np.count_nonzero(bundle_overlap)
            overreach = np.count_nonzero(bundle_overreach)
            lacking = np.count_nonzero(bundle_lacking)

            # Endpoints coverage
            endpoints_overlap = gt_masks[i] * current_tc_endpoints_voxels
            endpoints_overreach = np.zeros(dimensions)
            endpoints_overreach[np.where(
                (gt_masks[i] == 0) &
                (current_tc_endpoints_voxels >= 1))] = 1
        else:
            tc_dice = None
            wpc_dice = None
            overlap = None
            overreach = None
            lacking = None
            endpoints_overlap = None
            endpoints_overreach = None

        tmp_dict = {
            "bundle": bundles_names[i],
            "tc_streamlines": len(current_tc_streamlines),
            "wpc_streamlines": len(current_wpc_streamlines),
            "tc_dice": tc_dice,
            "wpc_dice": wpc_dice,
            "tc_bundle_overlap": overlap,
            "tc_bundle_overreach": overreach,
            "tc_bundle_lacking": lacking,
            "tc_bundle_overlap_PCT": overlap / (overlap + lacking),
            "tc_endpoints_overlap": np.count_nonzero(
                endpoints_overlap),
            "tc_endpoints_overreach": np.count_nonzero(
                endpoints_overreach)}

        tractogram_overlap += tmp_dict["tc_bundle_overlap_PCT"]
        tc_bundle_wise_dict.update({bundles_names[i]: tmp_dict})

    if args.compute_fc:
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
    else:
        bundle_wise_dict = {
            "true_connections": tc_bundle_wise_dict,
            "false_connections": "Not computed"
        }

    final_results.update({
        "bundle_wise": bundle_wise_dict,
        "tractogram_overlap": tractogram_overlap / nb_bundles
    })

    return final_results


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.gt_config)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)
    os.makedirs(os.path.join(args.out_dir, 'segmented_VB'))
    if args.compute_fc:
        os.makedirs(os.path.join(args.out_dir, 'segmented_IB'))
    path_duplicates = os.path.join(args.out_dir, 'segmented_conflicts')

    # -----------
    # Preparation
    # -----------
    # Read the config file
    (bundles_names, gt_masks_files, limits_masks_files, roi_options,
     lengths, angles) = read_config_file(args)

    # Find all masks to be loaded.
    all_rois = list(itertools.chain(
        *[list(roi_options[b].values()) for b in roi_options]))
    all_rois = list(dict.fromkeys(all_rois))  # Remove duplicates

    # Verify options
    assert_inputs_exist(parser, gt_masks_files + limits_masks_files +
                        all_rois + [args.in_tractogram])

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Loading tractogram.")
    _, ext = os.path.splitext(args.in_tractogram)
    sft = load_tractogram_with_reference(
        parser, args, args.in_tractogram, bbox_check=False)

    if args.remove_invalid:
        sft.remove_invalid_streamlines()

    logging.info("Verifying compatibility of tractogram with gt_masks and "
                 "limits_masks.")
    all_masks = gt_masks_files + limits_masks_files
    all_masks = list(dict.fromkeys(all_masks))  # Removes duplicates
    _verify_compatibility_with_bundles(sft, all_masks, parser, args)

    logging.info("Loading and/or computing ground-truth masks and limits "
                 "masks.")
    gt_masks, _, affine, dimensions, = \
        compute_masks(gt_masks_files, parser, args)
    limits_masks, limits_inv_masks, _, _, = \
        compute_masks(limits_masks_files, parser, args)

    logging.info("Extracting ground-truth head and tail masks.")
    gt_tails, gt_heads = compute_endpoint_masks(
        roi_options, affine, dimensions, args.out_dir)

    # Update all_rois, remove duplicates
    all_rois = gt_tails + gt_heads
    all_rois = list(dict.fromkeys(all_rois))  # Removes duplicates

    logging.info("Verifying tractogram compatibility with endpoint ROIs.")
    for file in all_rois:
        compatible = is_header_compatible(sft, file)
        if not compatible:
            parser.error("Input tractogram incompatible with {}".format(file))

    # True connections
    logging.info("Scoring true connections")
    tc_sft_list, tc_ids_list, wpc_ids_list, bundles_stats = \
        compute_true_connections_all_bundles(
            gt_tails, gt_heads, sft, lengths, angles, bundles_names,
            limits_inv_masks, args, ext, path_duplicates)

    # Cleaning wpc.
    logging.info("Verifying wpc")
    wpc_sft_list, bundles_stats = clean_wpc_all_bundles(
        wpc_ids_list, sft, bundles_names, args, ext, tc_ids_list,
        bundles_stats)

    # False connections
    if args.compute_fc:
        logging.info("Scoring false connections")
        # Keep all possible combinations
        all_rois = sorted(all_rois)
        comb_filename = list(itertools.combinations(all_rois, r=2))

        # Remove the true connections from all combinations, leaving only
        # false connections
        tc_filenames = list(zip(gt_tails, gt_heads))
        for tc_f in tc_filenames:
            tc_f = tuple(sorted(tc_f))
            comb_filename.remove(tc_f)
        fc_sft_list, fc_ids_list = compute_false_connections_all_bundles(
            comb_filename, sft, args)
    else:
        fc_ids_list = []
        fc_sft_list = []
        comb_filename = None

    all_tc_ids = np.unique(list(itertools.chain(*tc_ids_list)))
    all_wpc_ids = np.unique(list(itertools.chain(*wpc_ids_list)))
    all_fc_ids = np.unique(list(itertools.chain(*fc_ids_list)))

    # No connections
    # (ids that are not tc, not wpc and not fc.)
    all_nc_ids = np.arange(len(sft))
    all_nc_ids = np.setdiff1d(all_nc_ids, all_tc_ids)
    all_nc_ids = np.setdiff1d(all_nc_ids, all_wpc_ids)
    all_nc_ids = np.setdiff1d(all_nc_ids, all_fc_ids)

    logging.info("The remaining {} / {} streamlines will be scored as nc."
                 .format(len(all_nc_ids), len(sft)))

    nc_sft = make_sft_from_ids(all_nc_ids, sft)
    if len(nc_sft) > 0 or not args.no_empty:
        save_tractogram(nc_sft, os.path.join(
            args.out_dir, "nc{}".format(ext)), bbox_valid_check=False)

    final_results = compute_tractometry(
        all_tc_ids, all_wpc_ids, all_fc_ids, all_nc_ids, tc_ids_list,
        wpc_ids_list, fc_ids_list, tc_sft_list, wpc_sft_list, fc_sft_list, sft,
        args, bundles_names, gt_masks, dimensions, comb_filename)
    logging.info("Final results saved in {}".format(args.out_dir))
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(final_results, f,
                  indent=args.indent,
                  sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
