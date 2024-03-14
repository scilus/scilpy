#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Scores input tractogram overall and bundlewise.

Outputs
-------

    - results.json: Contains a full tractometry report.
    - processing_stats.json: Contains information on the segmentation of
    bundles (ex: the number of wpc per criteria).
    - Splits the input tractogram into
        segmented_VB/*_VS.trk.
        segmented_IB/*_*_IC.trk   (if args.compute_ic)
        segmented_WPC/*_wpc.trk  (if args.save_wpc_separately)
        IS.trk     OR      NC.trk  (if args.compute_ic)

By default, if a streamline fits in many bundles, it will be included in every
one. This means a streamline may be a VS for a bundle and an IS for
(potentially many) others. If you want to assign each streamline to at most one
bundle, use the `--unique` flag.

Config file
-----------

The config file needs to be a json containing a dict of the ground-truth
bundles as keys. The value for each bundle is itself a dictionnary with:

Mandatory:
    - endpoints OR [head AND tail]: filename for the endpoints ROI.
        If 'enpoints' is used, we will automatically separate the mask into two
        ROIs, acting as head and tail. Quality check is strongly recommended.

Optional:
    Concerning metrics:
    - gt_mask: expected result. OL and OR metrics will be computed from this.*

    Concerning inclusion criteria (other streamlines will be WPC):
    - all_mask: ROI serving as "all" criteria: to be included in the bundle,
        ALL points of a streamline must be inside the mask.*
    - any_mask: ROI serving as "any" criteria: streamlines
        must touch that mask in at least one point ("any" point) to be included
        in the bundle.
    - angle: angle criteria. Streamlines containing loops and sharp turns above
        given angle will be rejected from the bundle.
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

Exemple config file:
{
  "Ground_truth_bundle_0": {
    "gt_mask": "PATH/bundle0.nii.gz",
    "angle": 300,
    "length": [140, 150],
    "endpoints": "PATH/file1.nii.gz"
  }
}
"""
import argparse
import json
import itertools
import logging
import numpy as np
import os

from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible

from scilpy.io.streamlines import (load_tractogram_with_reference,
                                   verify_compatibility_with_reference_sft)
from scilpy.io.utils import (add_bbox_arg,
                             add_overwrite_arg,
                             add_json_args,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             assert_outputs_exist)
from scilpy.segment.tractogram_from_roi import (compute_masks_from_bundles,
                                                compute_endpoint_masks,
                                                segment_tractogram_from_roi)
from scilpy.tractanalysis.scoring import compute_tractometry
from scilpy.tractanalysis.scoring import __doc__ as tractometry_description

def_len = [0, np.inf]


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__ + tractometry_description,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("in_tractogram",
                   help="Input tractogram to score")
    p.add_argument("gt_config",
                   help=".json dict configured as specified above.")
    p.add_argument("out_dir",
                   help="Output directory for the resulting segmented "
                        "bundles.")
    p.add_argument("--json_prefix", metavar='p', default='',
                   help="Prefix of the two output json files. Ex: 'study_x_'."
                        "Files will be saved inside out_dir.\n"
                        "Suffixes will be 'processing_stats.json' and "
                        "'results.json'.")

    g = p.add_argument_group("Additions to gt_config")
    g.add_argument("--gt_dir", metavar='DIR',
                   help="Root path of the ground truth files listed in the "
                        "gt_config. \nIf not set, filenames in the config "
                        "file are considered \nas absolute paths.")
    g.add_argument("--use_gt_masks_as_all_masks", action='store_true',
                   help="If set, the gt_config's 'gt_mask' will also be used "
                        "as\n'all_mask' for each bundle. Note that this "
                        "means the\nOR will necessarily be 0.")

    g = p.add_argument_group("Preprocessing")
    g.add_argument("--dilate_endpoints",
                   metavar="NB_PASS", default=0, type=int,
                   help="Dilate endpoint masks n-times. Default: 0.")
    g.add_argument("--remove_invalid", action="store_true",
                   help="Remove invalid streamlines before scoring.")

    g = p.add_argument_group("Tractometry choices")
    g.add_argument("--save_wpc_separately", action='store_true',
                   help="If set, streamlines rejected from VC based on the "
                        "config\nfile criteria will be saved separately from "
                        "IS (and IC)\nin one file *_wpc.tck per bundle.")
    g.add_argument("--compute_ic", action='store_true',
                   help="If set, IS are split into NC + IC, where IC are "
                        "computed as one bundle per\npair of ROI not "
                        "belonging to a true connection, named\n*_*_IC.tck.")
    g.add_argument("--unique", action='store_true',
                   help="If set, streamlines are assigned to the first bundle"
                        " they fit in and not to all.")
    g.add_argument("--remove_wpc_belonging_to_another_bundle",
                   action='store_true',
                   help="If set, WPC actually belonging to any VB (in the \n"
                        "case of overlapping ROIs) will be removed\n"
                        "from the WPC classification.")

    p.add_argument("--no_empty", action='store_true',
                   help='Do not write file if there is no streamline.')

    add_json_args(p)
    add_bbox_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def load_and_verify_everything(parser, args):
    """
    - Reads the config file
    - Loads the masks / sft
        - If endpoints were given instead of head + tail, separate into two
          sub-rois.
    - Verifies compatibility
    """
    args.json_prefix = os.path.join(args.out_dir, args.json_prefix)
    json_outputs = [args.json_prefix + 'processing_stats.json',
                    args.json_prefix + 'results.json']
    assert_inputs_exist(parser, args.gt_config, args.reference)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)
    assert_outputs_exist(parser, args, json_outputs)
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
    list_masks_files_r = list(itertools.chain(
        *[list(roi_option.values()) for roi_option in roi_options]))
    list_masks_files_o = gt_masks_files + all_masks_files + any_masks_files
    # (This removes duplicates:)
    list_masks_files_r = list(dict.fromkeys(list_masks_files_r))
    list_masks_files_o = list(dict.fromkeys(list_masks_files_o))

    # Verify options
    assert_inputs_exist(parser, list_masks_files_r + [args.in_tractogram],
                        list_masks_files_o)

    logging.info("Loading tractogram.")
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    _, dimensions, _, _ = sft.space_attributes

    if args.remove_invalid:
        sft.remove_invalid_streamlines()

    logging.info("Verifying compatibility of tractogram with gt_masks, "
                 "all_masks, any_masks, ROI masks")
    verify_compatibility_with_reference_sft(
        sft, list_masks_files_r + list_masks_files_o, parser, args)

    logging.info("Loading and/or computing ground-truth masks, limits "
                 "masks and any_masks.")
    gt_masks = compute_masks_from_bundles(gt_masks_files, parser, args)
    inv_all_masks = compute_masks_from_bundles(all_masks_files, parser, args,
                                               inverse_mask=True)
    any_masks = compute_masks_from_bundles(any_masks_files, parser, args)

    logging.info("Extracting ground-truth head and tail masks.")
    gt_tails, gt_heads = compute_endpoint_masks(roi_options, args.out_dir)

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
            inv_all_masks, gt_masks, any_masks, dimensions, json_outputs)


def read_config_file(args):
    """
    Reads the gt_config file and returns:

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
    show_warning_gt = False

    with open(args.gt_config, "r") as json_file:
        config = json.load(json_file)

        bundles = list(config.keys())
        for bundle in bundles:
            bundle_config = config[bundle]

            if 'gt_mask' not in bundle_config:
                show_warning_gt = True
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

    if show_warning_gt:
        logging.info(
            "At least one bundle had no gt_mask. Some tractometry metrics "
            "won't be computed (OR, OL) for these bundles.")

    return (bundles, gt_masks, all_masks, any_masks, roi_options,
            lengths, angles, orientation_lengths, abs_orientation_lengths)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Load
    (gt_tails, gt_heads, sft, bundle_names, list_rois, bundle_lengths, angles,
     orientation_lengths, abs_orientation_lengths, inv_all_masks, gt_masks,
     any_masks, dimensions,
     json_outputs) = load_and_verify_everything(parser, args)

    # Segment VB, WPC, IB
    (vb_sft_list, wpc_sft_list, ib_sft_list, nc_sft,
     ib_names, bundle_stats) = segment_tractogram_from_roi(
        sft, gt_tails, gt_heads, bundle_names, bundle_lengths, angles,
        orientation_lengths, abs_orientation_lengths, inv_all_masks, any_masks,
        list_rois, args)

    # Save results
    with open(json_outputs[0], "w") as f:
        json.dump(bundle_stats, f, indent=args.indent,
                  sort_keys=args.sort_keys)

    logging.info("Final segmented bundles will be saved in {}"
                 .format(args.out_dir))
    for i in range(len(bundle_names)):
        if len(vb_sft_list[i]) > 0 or not args.no_empty:
            filename = "segmented_VB/{}_VS.trk".format(bundle_names[i])
            save_tractogram(vb_sft_list[i],
                            os.path.join(args.out_dir, filename),
                            bbox_valid_check=args.bbox_check)
        if (args.save_wpc_separately and wpc_sft_list[i] is not None
                and (len(wpc_sft_list[i]) > 0 or not args.no_empty)):
            filename = "segmented_WPC/{}_wpc.trk".format(bundle_names[i])
            save_tractogram(wpc_sft_list[i],
                            os.path.join(args.out_dir, filename),
                            bbox_valid_check=args.bbox_check)
    for i in range(len(ib_sft_list)):
        if len(ib_sft_list[i]) > 0 or not args.no_empty:
            file = "segmented_IB/{}_IC.trk".format(ib_names[i])
            save_tractogram(ib_sft_list[i], os.path.join(args.out_dir, file),
                            bbox_valid_check=args.bbox_check)

    # Tractometry on bundles
    final_results = compute_tractometry(
        vb_sft_list, wpc_sft_list, ib_sft_list, nc_sft,
        args, bundle_names, gt_masks, dimensions, ib_names)

    logging.info("Final scores will be saved in {}".format(json_outputs[1]))
    final_results.update({"tractogram_filename": str(args.in_tractogram)})
    with open(json_outputs[1], "w") as f:
        json.dump(final_results, f, indent=args.indent,
                  sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
