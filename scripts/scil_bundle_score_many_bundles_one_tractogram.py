#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This script is intended to score all bundles from a single tractogram. Each
valid bundle is compared to its ground truth.
Ex: It was used for the ISMRM 2015 Challenge scoring.

See also scil_bundle_score_same_bundle_many_segmentations.py to score many
versions of a same bundle, compared to ONE ground truth / gold standard.

This script is the second part of script scil_score_tractogram, which also
segments the wholebrain tractogram into bundles first.

Here we suppose that the bundles are already segmented and saved as follows:
    main_dir/
        segmented_VB/*_VS.trk.
        segmented_IB/*_*_IC.trk   (optional)
        segmented_WPC/*_wpc.trk  (optional)
        IS.trk  OR  NC.trk  (if segmented_IB is present)

Config file
-----------
The config file needs to be a json containing a dict of the ground-truth
bundles as keys. The value for each bundle is itself a dictionnary with:

    - gt_mask: expected result. OL and OR metrics will be computed from this.*

* Files must be .tck, .trk, .nii or .nii.gz. If it is a tractogram, a mask will
be created. If it is a nifti file, it will be considered to be a mask.

Exemple config file:
{
  "Ground_truth_bundle_0": {
    "gt_mask": "PATH/bundle0.nii.gz",
  }
}

Formerly: scil_score_bundles.py
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
                             assert_outputs_exist)
from scilpy.segment.tractogram_from_roi import compute_masks_from_bundles
from scilpy.tractanalysis.scoring import compute_tractometry
from scilpy.tractanalysis.scoring import __doc__ as tractometry_description


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__ + tractometry_description,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("gt_config",
                   help=".json dict configured as specified above.")
    p.add_argument("bundles_dir",
                   help="Directory containing all bundles.\n"
                        "(Ex: Output directory for scil_score_tractogram).\n"
                        "It is expected to contain a file IS.trk and \n"
                        "files segmented_VB/*_VS.trk, with, possibly, files \n"
                        "segmented_WPC/*_wpc.trk and segmented_IC/")
    p.add_argument("--json_prefix", metavar='p', default='',
                   help="Prefix of the output json file. Ex: 'study_x_'.\n"
                        "Suffix will be results.json. File will be saved "
                        "inside bundles_dir.\n")

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

    assert_inputs_exist(parser, args.gt_config, args.reference)
    if not os.path.isdir(args.bundles_dir):
        parser.error("Bundles dir ({}) does not exist."
                     .format(args.bundles_dir))

    args.json_prefix = os.path.join(args.bundles_dir, args.json_prefix)
    json_output = args.json_prefix + 'results.json'
    assert_outputs_exist(parser, args, json_output)

    # Read the config file
    bundle_names, gt_masks_files = read_config_file(args)

    # Not verifying compatibility of every mask with every sub-bundles...
    # Was probably done during segmentation.

    # Verify file organization
    # VB
    vb_path = os.path.join(args.bundles_dir, 'segmented_VB')
    if not os.path.isdir(vb_path):
        parser.error("We expect bundles_dir to contain a segmented_VB dir.\n"
                     "Read the help for more information.")
    # WPC
    wpc_path = os.path.join(args.bundles_dir, 'segmented_wpc')
    if not os.path.isdir(wpc_path):
        wpc_path = None
    # IC
    ib_path = os.path.join(args.bundles_dir, 'segmented_IB')
    nc_filename = os.path.join(args.bundles_dir, 'NC.trk')
    if not os.path.isdir(ib_path):  # --compute_ic was not used during segment.
        ib_path = None
        nc_filename = os.path.join(args.bundles_dir, 'IS.trk')
    if not os.path.isfile(nc_filename):
        logging.info("We expect bundles_dir to contain either a "
                     "segmented_IB dir together with a file NC.trk,"
                     "or a file IS.trk but neither was found.\n"
                     "Are you scoring a perfect tractogram?\n"
                     "Else, read the help for more information.")
        nc_filename = None

    # Now loading everything
    # Load gt masks
    logging.info("Loading and/or computing ground-truth masks.")
    gt_masks = compute_masks_from_bundles(gt_masks_files, parser, args)

    ref_sft = None

    # Load valid bundles
    vb_sft_list = []
    logging.info("Loading valid bundles")
    for bundle in bundle_names:
        vb_name = os.path.join(vb_path, bundle + '_VS.trk')
        if os.path.isfile(vb_name):
            sft = load_tractogram(vb_name, 'same',
                                  bbox_valid_check=args.bbox_check)
            vb_sft_list.append(sft)
            if ref_sft is None:
                ref_sft = sft
        else:
            logging.debug("Bundle {} was not found!".format(bundle))
            vb_sft_list.append([])  # nb streamlines will be len([]) = 0.

    # Load wpc bundles
    wpc_sft_list = []
    if wpc_path is not None:
        logging.info("Loading WPC bundles")
        for bundle in glob.glob(wpc_path + '/*'):
            sft = load_tractogram(bundle, 'same',
                                  bbox_valid_check=args.bbox_check)
            wpc_sft_list.append(sft)
            if ref_sft is None:
                ref_sft = sft
    else:
        logging.info("Did not find any WPC bundles.")

    # Load invalid bundles
    ib_sft_list = []
    ib_names = []
    if ib_path is not None:
        logging.info("Loading invalid bundles")
        for bundle in glob.glob(ib_path + '/*'):
            ib_names.append(os.path.basename(bundle))
            sft = load_tractogram(bundle, 'same',
                                  bbox_valid_check=args.bbox_check)
            ib_sft_list.append(sft)
            if ref_sft is None:
                ref_sft = sft
    else:
        logging.info("Did not find any invalid bundles.")

    # Load either NC or IS
    if nc_filename is not None:
        nc_sft = load_tractogram(nc_filename, 'same',
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
        The gt_mask filenames per bundle (None if not set) (used for
        tractometry statistics).
    """
    gt_masks = []
    with open(args.gt_config, "r") as json_file:
        config = json.load(json_file)

        bundles = list(config.keys())
        for bundle in bundles:
            bundle_config = config[bundle]
            if 'gt_mask' not in bundle_config:
                raise ValueError("Tracometry cannot be computed if no gt_mask "
                                 "is given.")
            if args.gt_dir:
                gt_masks.append(os.path.join(args.gt_dir,
                                             bundle_config['gt_mask']))
            else:
                gt_masks.append(bundle_config['gt_mask'])

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
        "bundles_dir": str(args.bundles_dir),
    })

    logging.info("Final results saved in {}".format(out_filename))
    with open(out_filename, "w") as f:
        json.dump(final_results, f, indent=args.indent,
                  sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
