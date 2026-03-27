#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This script is intended to score all bundles from a single tractogram. Each
valid bundle is compared to its ground truth.
Ex: It was used for the ISMRM 2015 Challenge scoring.

Our bundle tractometry analysis scripts
---------------------------------------
- scil_bundle_pairwise_comparison:
    - can compare any pair (any combo) from the input list.
    - can compare the input list against a single file (--single_compare)
- scil_bundle_score_many_bundles_one_tractogram:
    - scores files against their individual ground truth.
- scil_bundle_score_same_bundle_many_segmentations:
    - compare many versions of a single bundle.
* If you have volumes associated to your bundles, the following script could be
of interest for you: scil_volume_pairwise_comparison

This script
-----------
We expect your tractogram to be already segmented into valid bundles.
See our scripts scil_tractogram_segment_with [...].
For usage in combination with scil_tractogram_segment_with_ROI_and_score, see
particular instructions below.

The easiest usage of this script requires:
    - Having all bundles to be scored in a single folder.
    - Creating a config file associating each input tractogram to its
      reference tractogram.
>> scil_bundle_score_many_bundles_one_tractogram config_file VB_folder

Additionnaly, this script can include, in the output json, the percentage of
invalid streamlines (IS) or invalid bundles, using --is, --ib, etc.

Config file
-----------
The config file needs to be a json containing a dict of the ground-truth
bundles as keys. The value for each bundle is itself a dictionnary with the
path to reference bundles. The reference names must be included in the
segmented bundles' names. Ex: VB_folder/segmented_CC_subjX.trk.

Example config file:
{
  "CC": "PATH/ref_grouth_truth_CC.nii.gz",
  "OR_L": "PATH/ref_grouth_truth_OR_L.nii.gz",
}
* Files must be .tck, .trk, .nii or .nii.gz. If it is a tractogram, a mask will
be created. If it is a nifti file, it will be considered to be a mask.

Usage as part 2 of segment_with_ROI
-----------------------------------
This script can be used as the second part of script
>> scil_tractogram_segment_with_ROI_and_score
Then, we suppose that the bundles are already segmented and saved as follows:
    root_dir/
        segmented_VB/*_VS.trk.
        segmented_IB/*_*_IC.trk   (optional)
        segmented_WPC/*_wpc.trk  (optional)
        IS.trk  OR  NC.trk  (if segmented_IB is present)
Use option --part2_ROI_segmentation, and, as main input folder, give the
root_dir instead of the VB_folder. We will automatically set VB_folder, --is,
--vb and --wpc to follow this organization. Also, use the same config file as
in scil_tractogram_segment_with_ROI_and_score:
{
  "Ground_truth_bundle_0": {
    "gt_mask": "PATH/bundle0.nii.gz",
  }
}

"""

import argparse
import glob
import json
import logging
import os

from dipy.io.streamline import load_tractogram

from scilpy.io.utils import (add_bbox_arg,
                             add_overwrite_arg,
                             add_json_args,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist, assert_inputs_dirs_exist,
                             assert_headers_compatible)
from scilpy.segment.tractogram_from_roi import compute_masks_from_bundles
from scilpy.tractanalysis.scoring import compute_tractometry
from scilpy.tractanalysis.scoring import __doc__ as tractometry_description
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__ + tractometry_description,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument("gt_config",
                   help=".json config file configured as specified above.")
    p.add_argument("VB_folder",
                   help="Directory containing all bundles to be scored.\n"
                        "(Ex: Output directory from "
                        "scil_tractogram_segment_with_ROI_and_score \nor "
                        "from scil_tractogram_segment_with_bundleseg).\n"
                        "It is expected to contain a file IS.trk and \n"
                        "files segmented_VB/*_VS.trk, with, possibly, files \n"
                        "segmented_WPC/*_wpc.trk and segmented_IC/")
    # Note. Can't name --invalid as --is. Created errors with the is keyword.
    p.add_argument("--invalid", metavar="IS.trk",
                   help="To include the percentage of invalid streamlines "
                        "include your \ninvalid streamlines here (or "
                        "no-connection streamlines).")
    p.add_argument("--ib", metavar="IB/",
                   help="To include the percentage of invalid streamlines in "
                        "each segmented invalid \nbundle, add the path to "
                        "your invalid bundles here.")
    p.add_argument("--wpc", metavar="WPC/",
                   help="To include the percentage of wrong path streamlines "
                        "for each valid bundle, \nadd the path to your WPC "
                        "bundles here.")
    p.add_argument("--json_prefix", metavar='p', default='',
                   help="Prefix of the output json file. Ex: 'study_x_'.\n"
                        "Suffix will be results.json. If the prefix does not "
                        "contain a directory, \nfile will be saved inside "
                        "your root directory.")
    p.add_argument("--part2_ROI_segmentation", action="store_true",
                   help="If set, configure everything to be used as part 2 of "
                        "script \nscil_tractogram_segment_with_ROI_and_score.")

    g = p.add_argument_group("Additions to gt_config")
    g.add_argument("--gt_dir", metavar='DIR',
                   help="Root path of the ground truth files listed in the "
                        "gt_config.\nIf not set, filenames in the config "
                        "file are considered \nas absolute paths.")

    add_json_args(p)
    add_reference_arg(p)
    add_bbox_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def load_and_verify_everything(parser, args):

    # Input organization depends on option part2_ROI_segmentation
    args.root_dir = args.VB_folder
    if args.part2_ROI_segmentation:
        if (args.invalid is not None or args.ib is not None or
             args.wpc is not None):
            parser.error("--part2_ROI_segmentation can't be used together "
                         "with other options.")
        args.VB_folder = os.path.join(args.root_dir, "segmented_VB")
        sub_dirs = [x[0] for x in os.walk(args.root_dir)]
        if 'segmented_IB' in sub_dirs:
            args.ib = os.path.join(args.root_dir, "segmented_IB")
            args.invalid = os.path.join(args.root_dir, "NC.trk")
        elif os.path.exists(os.path.join(args.root_dir, "IS.trk")):
            args.invalid = os.path.join(args.root_dir, "IS.trk")
        if 'segmented_WPC' in sub_dirs:
            args.wpc = os.path.join(args.root_dir, "segmented_WPC")

    assert_inputs_exist(parser, args.gt_config,
                        [args.reference, args.invalid])
    assert_inputs_dirs_exist(parser, args.VB_folder,
                             [args.ib, args.wpc])

    # Outputs:
    _path, _name = os.path.split(args.json_prefix)
    if _path == '':
        args.json_prefix = os.path.join(args.root_dir, args.json_prefix)
    json_output = args.json_prefix + 'results.json'
    assert_outputs_exist(parser, args, json_output)

    # Read the config file
    bundle_names, gt_masks_files = read_config_file(args)

    # Check that all bundles are associated to a unique VB
    all_vb_files = glob.glob(args.VB_folder + '/*')
    ordered_vb_files = []
    for bname in bundle_names:
        vb_files = [filename for filename in all_vb_files if bname in filename]
        if len(vb_files) > 1:
            parser.error("The bundle name {} was found in more than one valid "
                         "bundles: {}".format(bname, vb_files))
        if len(vb_files) == 0:
            logging.info("The bundle {} was not found in the name of any "
                            "file inside the VB folder {}. Score will be 0."
                            .format(bname, args.VB_folder))
            ordered_vb_files.append(None)
        else:
            ordered_vb_files.append(vb_files[0])

    # Not checking here IB and WPC files.
    assert_headers_compatible(parser, gt_masks_files, ordered_vb_files,
                              reference=args.reference)

    # ----------------------
    # Now loading everything
    # ----------------------
    if args.reference is None:
        args.reference = 'same'

    # Load gt masks
    logging.info("Loading and/or computing ground-truth masks.")
    gt_masks = compute_masks_from_bundles(gt_masks_files, parser, args)

    ref_sft = None  # Will be the first sft

    # Load valid bundles
    vb_sft_list = []
    logging.info("Loading valid bundles")
    for bname, vb_file in zip(bundle_names, ordered_vb_files):
        if vb_file is not None:
            sft = load_tractogram(vb_file, args.reference,
                                  bbox_valid_check=args.bbox_check)
            vb_sft_list.append(sft)
            if ref_sft is None:
                ref_sft = sft
        else:
            logging.debug("Bundle {} was not found!".format(bname))
            vb_sft_list.append([])  # nb streamlines will be len([]) = 0.

    # Load wpc bundles
    wpc_sft_list = []
    if args.wpc is not None:
        logging.info("Loading WPC bundles")
        for bname in glob.glob(args.wpc + '/*'):
            sft = load_tractogram(bname, args.reference,
                                  bbox_valid_check=args.bbox_check)
            wpc_sft_list.append(sft)
            if ref_sft is None:
                ref_sft = sft
    else:
        logging.info("Did not find any WPC bundles.")

    # Load invalid bundles
    ib_sft_list = []
    ib_names = []
    if args.ib is not None:
        logging.info("Loading invalid bundles")
        for bname in glob.glob(args.ib + '/*'):
            ib_names.append(os.path.basename(bname))
            sft = load_tractogram(bname, args.reference,
                                  bbox_valid_check=args.bbox_check)
            ib_sft_list.append(sft)
            if ref_sft is None:
                ref_sft = sft
    else:
        logging.info("Did not find any invalid bundles.")

    # Load either NC or IS
    if args.invalid is not None:
        nc_sft = load_tractogram(args.invalid, args.reference,
                                 bbox_valid_check=args.bbox_check)
        ref_sft = nc_sft
    else:
        nc_sft = None

    if ref_sft is None:
        print("No bundle found (VB, WPC, IS, NC). Stopping.")
        exit(0)
    _, dimensions, _, _ = ref_sft.space_attributes

    return (bundle_names, gt_masks, dimensions, vb_sft_list, wpc_sft_list,
            ib_sft_list, nc_sft, ib_names, json_output)


def read_config_file(args):
    """
    Reads the gt_config file and returns:

    Returns
    -------
    bundles: List
        The names of each bundle.
    gt_masks: List
        The gt_mask filenames per bundle.
    """
    gt_masks = []
    with open(args.gt_config, "r") as json_file:
        config = json.load(json_file)

        bundles = list(config.keys())

        if args.part2_ROI_segmentation:
            for bundle in bundles:
                bundle_config = config[bundle]
                if 'gt_mask' not in bundle_config:
                    raise ValueError("Tractometry cannot be computed if no "
                                     "gt_mask is given.")
                if args.gt_dir:
                    gt_masks.append(os.path.join(args.gt_dir,
                                                 bundle_config['gt_mask']))
                else:
                    gt_masks.append(bundle_config['gt_mask'])
        else:
            for bundle in bundles:
                bundle_path = config[bundle]
                if args.gt_dir:
                    gt_masks.append(os.path.join(args.gt_dir, bundle_path))
                else:
                    gt_masks.append(bundle_path)
    return bundles, gt_masks


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    (bundle_names, gt_masks, dimensions,
     vb_sft_list, wpc_sft_list, ib_sft_list, nc_sft,
     ib_names, out_filename) = load_and_verify_everything(parser, args)

    args.compute_ic = True if len(ib_sft_list) > 0 else False
    args.save_wpc_separately = True if len(wpc_sft_list) > 0 else False

    # Tractometry
    final_results = compute_tractometry(
        vb_sft_list, wpc_sft_list, ib_sft_list, nc_sft,
        args, bundle_names, gt_masks, dimensions, ib_names)
    final_results.update({
        "root_dir": str(args.root_dir),
    })

    logging.info("Final results saved in {}".format(out_filename))
    with open(out_filename, "w") as f:
        json.dump(final_results, f, indent=args.indent,
                  sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
