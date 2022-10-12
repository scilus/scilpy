#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Scores input tractogram overall and bundlewise. Outputs a results.json
containing a full report, a processing_stats.json containing information on the
formation of bundles (ex: the number of wpc per criteria), and splits the input
tractogram into one file per bundle : *_VS.tck. Remaining streamlines are
combined in a IS.tck file.

By default, if a streamline fits in many bundles, it will be included in every
one. This means a streamline may be a VS for a bundle and an IS for
(potentially many) others. If you want to assign each streamline to at most one
bundle, use the `--unique` flag.

Definitions:
    In terms of number of streamlines:
        Computed by default:
        - VS: valid streamlines, belonging to a bundle (i.e. respecting all the
            criteria for that bundle; endpoints, limit_mask, gt_mask.).
        - WPC: wrong path connections, streamlines connecting correct ROIs but
            not respecting the other criteria for that bundle. The WPC
            statistics are saved into processing_stats.json, but the results
            are only saved if specified in the options. Else, they are merged
            back with the IS.

            **WPC are only computed if "limits masks" are provided.** Else,
            they are considered VS.

        - IS: invalid streamlines. All other streamlines.

        Optional:
        - IC: invalid connections, streamlines joining an incorrect combination
            of ROIs. Use carefully, quality depends on the quality of your ROIs
            and no analysis is done on the shape of the streamlines.
        - NC: no connections. Invalid streamlines minus invalid connections.

    In terms of number of voxels:
    - OL : percentage of ground truth voxels containing VS streamline(s).
    - OR/ORn: percentage of voxels containing VS streamline(s) when it
        shouldn't. We compute two versions of the overreach:
        OR = % of the recovered bundle. Values range between 0 and 100%. Values
           are not defined with we recovered no streamline for a bundle, but we
           set the OR to 0 in that case.
        ORn = % of the ground truth bundle. Values could be higher than 100%.

Config file:
    The config file needs to be a json containing a dict of the ground-truth
    bundles as keys. The value for each bundle is itself a dictionnary with:

    Mandatory:
    - endpoints OR [head AND tail]: filename for the endpoints ROI.
        If 'enpoints' is used, we will automatically separate the mask into
        two ROIs, acting as head and tail. Quality check is strongly
        recommended.

    Optional:
        Concerning metrics:
        - gt_mask: expected result. OL and OR metrics will be computed from
            this.*

        Concerning inclusion criteria (other streamlines will be WPC):
        - all_mask: ROI serving as "all" criteria: to be included in the
            bundle, ALL points of a streamline must be inside the mask.*
        - any_mask: ROI serving as "any" criteria: streamlines
            must touch that mask in at least one point ("any" point) to be
            included in the bundle.
        - angle: angle criteria. Streamlines containing loops and sharp turns
            above given angle will be rejected from the bundle.
        - length: maximum and minimum lengths per bundle.
        - length_x / length_x_abs: maximum and minimum total distance in the x
            direction (i.e. first coordinate).**
        - length_y / length_y_abs: maximum and minimum total distance in the y
            direction (i.e. second coordinate).**
        - length_z / length_z_abs: maximum and minimum total distance in the z
            direction (i.e. third coordinate).**

* Files must be .tck, .trk, .nii or .nii.gz. If it is a tractogram, a mask will
be created. If it is a nifti file, it will be considered to be a mask.
** With absolute values: coming back on yourself will contribute to the total
distance instead of cancelling it.

Ex 1:
{
  "Ground_truth_bundle_0": {
    "gt_mask": "PATH/bundle0.nii.gz",
    "angle": 300,
    "length": [140, 150],
    "endpoints": "PATH/file1.nii.gz"
  }
}

Ex 2:
(with options --gt_dir PATH)
{
  "Ground_truth_bundle_1": {
    "gt_mask": "masks/bundle1.trk"
    "head": 'roi/file2',
    "tail": 'roi/file3',
    "all_mask": "masks/general_envelope_bundle_1.nii.gz"
  }
}

"""
import argparse
import json
import itertools
import logging
import numpy as np
import os

from dipy.io.utils import is_header_compatible

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_json_args,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             verify_compatibility_with_reference_sft)
from scilpy.segment.tractogram_from_roi import segment_tractogram_from_roi, \
    compute_masks, compute_endpoint_masks
from scilpy.tractanalysis.scoring import compute_tractometry

def_len = [0, np.inf]


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("in_tractogram",
                   help="Input tractogram to score")
    p.add_argument("gt_config",
                   help=".json dict configured as specified above.")
    p.add_argument("out_dir",
                   help="Output directory.")

    g = p.add_argument_group("Additions to gt_config")
    p.add_argument("--gt_dir", metavar='DIR',
                   help="Root path of the ground truth files listed in the "
                        "gt_config.\n If not set, filenames in the config "
                        "file are considered\n as complete paths.")
    g.add_argument("--use_gt_masks_as_all_masks", action='store_true',
                   help="If set, the gt_config's 'gt_mask' will also be used "
                        "as\n'all_mask' for each bundle. Note that this "
                        "means the\nOR will necessarily be 0.")

    g = p.add_argument_group("Preprocessing")
    g.add_argument("--dilate_endpoints",
                   metavar="NB_PASS", default=0, type=int,
                   help="Dilate inclusion masks n-times. Default: 0.")
    g.add_argument("--remove_invalid", action="store_true",
                   help="Remove invalid streamlines before scoring.")

    g = p.add_argument_group("Tractometry choices")
    g.add_argument("--save_wpc_separately", action='store_true',
                   help="If set, streamlines rejected from VC based on the "
                        "config\nfile criteria will be saved separately from "
                        "IS (and IC)\nin one file *_WPC.tck per bundle.")
    g.add_argument("--compute_ic", action='store_true',
                   help="If set, IS are split into NC + IC, where IC are "
                        "computed as one bundle per\npair of ROI not "
                        "belonging to a true connection, named\n*_*_IC.tck.")
    g.add_argument("--unique", action='store_true',
                   help="If set, streamlines are assigned to the first bundle"
                        " they fit in and not to all.")
    g.add_argument("--remove_wpc_belonging_to_another_bundle",
                   action='store_true',
                   help="If set, WPC actually belonging to VC (from another "
                        "bundle,\nof course; in the case of overlapping ROIs) "
                        "will be removed\nfrom the WPC classification.")

    p.add_argument("--no_empty", action='store_true',
                   help='Do not write file if there is no streamline.')

    add_json_args(p)
    add_overwrite_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)

    return p


def load_and_verify_everything(parser, args):
    """
    - Reads the config file
    - Loads the masks / sft
        - If endpoints were given instead of head + tail, separate into two
          sub-rois.
    - Verifies compatibility
    """
    assert_inputs_exist(parser, args.gt_config)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)
    os.makedirs(os.path.join(args.out_dir, 'segmented_VB'))
    if args.compute_ic:
        os.makedirs(os.path.join(args.out_dir, 'segmented_IB'))
    if args.save_wpc_separately:
        os.makedirs(os.path.join(args.out_dir, 'segmented_WPC'))

    # Read the config file
    (bundle_names, gt_masks_files, all_masks_files, any_masks_files,
     roi_options, lengths, angles, orientation_lengths,
     abs_orientation_lengths) = read_config_file(args)

    # Find every mandatory mask to be loaded
    list_masks_files = list(itertools.chain(
        *[list(roi_option.values()) for roi_option in roi_options]))
    # (This removes duplicates:)
    list_masks_files = list(dict.fromkeys(list_masks_files))

    # Verify options
    assert_inputs_exist(parser, list_masks_files + [args.in_tractogram],
                        gt_masks_files + all_masks_files +
                        any_masks_files)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Loading tractogram.")
    sft = load_tractogram_with_reference(
        parser, args, args.in_tractogram, bbox_check=False)

    if args.remove_invalid:
        sft.remove_invalid_streamlines()

    logging.info("Verifying compatibility of tractogram with gt_masks, "
                 "all_masks and any_masks")
    list_masks_files = gt_masks_files + all_masks_files + any_masks_files
    # Removing duplicates:
    list_masks_files = list(dict.fromkeys(list_masks_files))
    verify_compatibility_with_reference_sft(sft, list_masks_files, parser,
                                            args)

    logging.info("Loading and/or computing ground-truth masks, limits "
                 "masks and any_masks.")
    gt_masks, _, affine, dimensions, = \
        compute_masks(gt_masks_files, parser, args)
    _, inv_all_masks, _, _, = \
        compute_masks(all_masks_files, parser, args)
    any_masks, _, _, _, = \
        compute_masks(any_masks_files, parser, args)

    logging.info("Extracting ground-truth head and tail masks.")
    gt_tails, gt_heads = compute_endpoint_masks(
        roi_options, affine, dimensions, args.out_dir)

    # Update the list of every ROI, remove duplicates
    list_rois = gt_tails + gt_heads
    list_rois = list(dict.fromkeys(list_rois))  # Removes duplicates

    logging.info("Verifying tractogram compatibility with endpoint ROIs.")
    for file in list_rois:
        compatible = is_header_compatible(sft, file)
        if not compatible:
            parser.error("Input tractogram incompatible with {}".format(file))

    return (gt_tails, gt_heads, sft, bundle_names, list_rois,
            lengths, angles, orientation_lengths, abs_orientation_lengths,
            inv_all_masks, gt_masks, any_masks, dimensions)


def read_config_file(args):
    """
    Read the gt_config file and returns:

    Returns
    -------
    bundles: List
        The names of each bundle.
    gt_masks: List
        The gt_mask filenames per bundle (None if not set) (used for
        tractometry statistics).
    all_masks: List
        The all_masks filenames per bundles (None if not set).
    any_masks: List
        The any_masks filenames per bundles (None if not set).
    roi_options: List
        The roi_option dict per bundle. Keys are 'gt_head', 'gt_tail' if
        they are set, else 'gt_endpoints'.
    angles: List
        The maximum angles per bundle (None if not set).
    lengths: List
        The [min max] lengths per bundle (None if not set).
    orientation_length: List
        The [[min_x, max_x], [min_y, max_y], [min_z, max_z]] per bundle.
        (None they are all not set).
    """
    angles = []
    lengths = []
    orientation_lengths = []
    abs_orientation_lengths = []
    gt_masks = []
    all_masks = []
    any_masks = []
    roi_options = []

    with open(args.gt_config, "r") as json_file:
        config = json.load(json_file)

        bundles = list(config.keys())
        for bundle in bundles:
            bundle_config = config[bundle]

            if 'gt_mask' not in bundle_config:
                logging.warning(
                    "No gt_mask set for bundle {}. Some tractometry metrics "
                    "won't be computed (OR, OL).".format(bundle))
            if 'endpoints' not in bundle_config and \
                    'head' not in bundle_config:
                raise ValueError(
                    "Bundle configuration for bundle {} misses 'endpoints' or "
                    "'head'/'tail'".format(bundle))

            angle = length = None
            length_x = length_y = length_z = None
            length_x_abs = length_y_abs = length_z_abs = None
            gt_mask = all_mask = any_mask = roi_option = None

            for key in bundle_config.keys():
                if key == 'angle':
                    angle = bundle_config['angle']
                elif key == 'length':
                    length = bundle_config['length']
                elif key == 'length_x':
                    length_x = bundle_config['length_x']
                elif key == 'length_y':
                    length_y = bundle_config['length_y']
                elif key == 'length_z':
                    length_z = bundle_config['length_z']
                elif key == 'length_x_abs':
                    length_x_abs = bundle_config['length_x_abs']
                elif key == 'length_y_abs':
                    length_y_abs = bundle_config['length_y_abs']
                elif key == 'length_z_abs':
                    length_z_abs = bundle_config['length_z_abs']
                elif key == 'gt_mask':
                    if args.gt_dir:
                        gt_mask = os.path.join(args.gt_dir,
                                               bundle_config['gt_mask'])
                    else:
                        gt_mask = bundle_config['gt_mask']

                    if args.use_gt_masks_as_all_masks:
                        all_mask = gt_mask
                elif key == 'all_mask':
                    if args.use_gt_masks_as_all_masks:
                        raise ValueError(
                            "With the option --use_gt_masks_as_all_masks, "
                            "you should not add any all_mask in the config "
                            "file.")
                    if args.gt_dir:
                        all_mask = os.path.join(args.gt_dir,
                                                bundle_config['all_mask'])
                    else:
                        all_mask = bundle_config['all_mask']
                elif key == 'endpoints':
                    if 'head' in bundle_config or 'tail' in bundle_config:
                        raise ValueError(
                            "Bundle {} has confusing keywords in the config "
                            "file. Please choose either endpoints OR "
                            "head/tail.".format(bundle))
                    if args.gt_dir:
                        endpoints = os.path.join(args.gt_dir,
                                                 bundle_config['endpoints'])
                    else:
                        endpoints = bundle_config['endpoints']
                    roi_option = {'gt_endpoints': endpoints}
                elif key == 'head':
                    if 'tail' not in bundle_config:
                        raise ValueError(
                            "You have provided the head for bundle {}, but "
                            "not the tail".format(bundle))
                    if args.gt_dir:
                        head = os.path.join(args.gt_dir, bundle_config['head'])
                        tail = os.path.join(args.gt_dir, bundle_config['tail'])
                    else:
                        head = bundle_config['head']
                        tail = bundle_config['tail']
                    roi_option = {'gt_head': head, 'gt_tail': tail}
                elif key == 'tail':
                    pass  # dealt with at head
                elif key == 'any_mask':
                    if args.gt_dir:
                        any_mask = os.path.join(
                            args.gt_dir, bundle_config['any_mask'])
                    else:
                        any_mask = bundle_config['any_mask']
                else:
                    raise ValueError("Unrecognized value {} in the config "
                                     "file for bundle {}".format(key, bundle))

            angles.append(angle)
            lengths.append(length)
            if length_x is None and length_y is None and length_z is None:
                orientation_lengths.append(None)
            else:
                orientation_lengths.append(
                    [length_x if length_x is not None else def_len,
                     length_y if length_y is not None else def_len,
                     length_z if length_z is not None else def_len])

            if length_x_abs is None and length_y_abs is None and \
                    length_z_abs is None:
                abs_orientation_lengths.append(None)
            else:
                abs_orientation_lengths.append(
                    [length_x_abs if length_x_abs is not None else def_len,
                     length_y_abs if length_y_abs is not None else def_len,
                     length_z_abs if length_z_abs is not None else def_len])
            gt_masks.append(gt_mask)
            all_masks.append(all_mask)
            any_masks.append(any_mask)
            roi_options.append(roi_option)

    return (bundles, gt_masks, all_masks, any_masks, roi_options,
            lengths, angles, orientation_lengths, abs_orientation_lengths)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load
    (gt_tails, gt_heads, sft, bundle_names, list_rois, bundle_lengths, angles,
     orientation_lengths, abs_orientation_lengths, inv_all_masks, gt_masks,
     any_masks, dimensions) = load_and_verify_everything(parser, args)

    sft.to_vox()

    # Segment VB, WPC, IB
    (all_vs_ids, all_wpc_ids, all_ic_ids, all_nc_ids,
     vs_ids_list, ic_ids_list, wpc_ids_list,
     vb_sft_list, wpc_sft_list, ib_sft_list,
     comb_filename) = segment_tractogram_from_roi(
        sft, gt_tails, gt_heads, bundle_names, bundle_lengths, angles,
        orientation_lengths, abs_orientation_lengths, inv_all_masks, any_masks,
        list_rois, args)

    # Tractometry on bundles
    final_results = compute_tractometry(
        all_vs_ids, all_wpc_ids, all_ic_ids, all_nc_ids,
        vs_ids_list, ic_ids_list, wpc_ids_list,
        vb_sft_list, wpc_sft_list, ib_sft_list, sft,
        args, bundle_names, gt_masks, dimensions, comb_filename)
    logging.info("Final results saved in {}".format(args.out_dir))
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=args.indent,
                  sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
