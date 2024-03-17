#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projects metrics onto the underlying voxels of a streamlines. This script can
project data from data_per_point (dpp) or data_per_streamline (dps) to maps.

You choose to project data from all points of the streamlines, or from the
endpoints only. The idea then is to visualize the cortical areas affected by
metrics (assuming streamlines start/end in the cortex).

See also scil_tractogram_project_map_to_streamlines.py for the reverse action.

How to the data is loaded:
    - From dps: uses the same value for each point of the streamline.
    - From dpp: one value per point.

How the data is used:
    1. Average all points of the streamline to get a mean value, set this value
       to all points.
    2. Average the two endpoints and get their mean value, set this value to
       all points.
    3. Keep each point individually.

How the data is projected to a map:
    A. Using each point.
    B. Using the endpoints only.

For more complex operations than the average per streamline, see
scil_tractogram_dpp_math.py.
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import (load_dpp_files_as_dpp,
                                   load_dps_files_as_dps,
                                   load_tractogram_with_reference)
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tractograms.dps_and_dpp_management import (
    convert_dps_to_dpp, perform_operation_dpp_to_dps, project_dpp_to_map)
from scilpy.utils.filenames import split_name_with_nii


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')
    p.add_argument('out_prefix',
                   help='Folder + prefix to save endpoints metric(s). We will '
                        'save \none nifti file per per dpp/dps key given.\n'
                        'Ex: my_path/subjX_bundleY_ with --use_dpp key1 '
                        'will output \nmy_path/subjX_bundleY_key1.nii.gz')

    p1 = p.add_argument_group(
        description='Where to get the statistics from. (Choose one)')
    p1 = p1.add_mutually_exclusive_group(required=True)
    p1.add_argument('--use_dps', metavar='key', nargs='+',
                    help='Use the data_per_streamline from the tractogram.\n'
                         'It must be a .trk')
    p1.add_argument('--use_dpp', metavar='key', nargs='+', default=[],
                    help='Use the data_per_point from the tractogram. \n'
                         'It must be a trk.')
    p1.add_argument('--load_dps', metavar='file', nargs='+', default=[],
                    help='Load data per streamline (scalar) .txt or .npy.\n'
                         'Must load an array with the right shape.')
    p1.add_argument('--load_dpp', metavar='file', nargs='+', default=[],
                    help='Load data per point (scalar) from .txt or .npy.\n'
                         'Must load an array with the right shape.')

    p2 = p.add_argument_group(description='Processing choices. (Choose one)')
    p2 = p2.add_mutually_exclusive_group(required=True)
    p2.add_argument('--mean_endpoints', action='store_true',
                    help="Uses one single value per streamline: the mean "
                         "of the two \nendpoints.")
    p2.add_argument('--mean_streamline', action='store_true',
                    help='Use one single value per streamline: '
                         'the mean of all \npoints of the streamline.')
    p2.add_argument('--point_by_point', action='store_true',
                    help="Directly project the streamlines values onto the "
                         "map.\n")

    p3 = p.add_argument_group(
        description='Where to send the statistics. (Choose one)')
    p3 = p3.add_mutually_exclusive_group(required=True)
    p3.add_argument('--to_endpoints', action='store_true',
                    help="Project metrics onto a mask of the endpoints.")
    p3.add_argument('--to_wm', action='store_true',
                    help='Project metrics into streamlines coverage.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _load_dpp_dps(args, parser, sft):
    # In call cases: only one of the values below can be set at the time.
    dps_to_use = None
    dpp_to_use = None

    # 1. With options --use_dps, --use_dpp: check that dps / dpp key is found.
    # 2. With options --load_dps, --load_dpp: Load them now to SFT, check that
    #    they fit with the data.
    if args.use_dps:
        dps_to_use = args.use_dps
        possible_dps = list(sft.data_per_streamline.keys())
        for key in args.use_dps:
            if key not in possible_dps:
                parser.error('DPS key not ({}) not found in your tractogram!'
                             .format(key))
    elif args.use_dpp:
        dpp_to_use = args.use_dpp
        possible_dpp = list(sft.data_per_point.keys())
        for key in args.use_dpp:
            if key not in possible_dpp:
                parser.error('DPP key ({}) not found in your tractogram!'
                             .format(key))
    elif args.load_dps:
        logging.info("Loading dps from file.")

        # It does not matter if we overwrite: Not saving the result sft.
        sft, dps_to_use = load_dps_files_as_dps(parser, args.load_dps, sft,
                                                overwrite=True)
    else:  # args.load_dpp:
        # Loading dpp for all points even if we won't use them all to make
        # sure that the loaded files have the correct shape.
        logging.info("Loading dpp from file")
        sft, dpp_to_use = load_dpp_files_as_dpp(parser, args.load_dpp, sft,
                                                overwrite=True)

    # Verify that we have singular values. (Ex, not colors)
    # Remove unused keys to save memory.
    all_keys = list(sft.data_per_point.keys())
    for key in all_keys:
        if dpp_to_use is not None and key in dpp_to_use:
            d0 = sft.data_per_point[key][0][0]
            if len(d0) > 1:
                raise ValueError(
                    "Expecting scalar values as data_per_point. Got data of "
                    "shape {} for key {}".format(d0.shape, key))
        else:
            del sft.data_per_point[key]

    all_keys = list(sft.data_per_streamline.keys())
    for key in all_keys:
        if dps_to_use is not None and key in dps_to_use:
            d0 = sft.data_per_streamline[key][0]
            if len(d0) > 1:
                raise ValueError(
                    "Expecting scalar values as data_per_streamline. Got data "
                    "of shape {} for key {}.".format(d0.shape, key))
        else:
            del sft.data_per_streamline[key]

    return sft, dps_to_use, dpp_to_use


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # -------- General checks ----------
    assert_inputs_exist(parser, [args.in_bundle],
                        args.load_dps + args.load_dpp + [args.reference])

    # Find all final output files (one per metric).
    if args.load_dps or args.load_dpp:
        files = args.load_dps or args.load_dpp
        metrics_names = []
        for file in files:
            # Prepare dpp key from filename.
            name = os.path.basename(file)
            name, ext = split_name_with_nii(name)
            metrics_names.append(name)
    else:
        metrics_names = args.use_dpp or args.use_dps
    out_files = [args.out_prefix + m + '.nii.gz' for m in metrics_names]
    assert_outputs_exist(parser, args, out_files)

    # -------- Load streamlines and checking compatibility ----------
    logging.info("Loading tractogram {}".format(args.in_bundle))
    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

    if len(sft.streamlines) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return

    # -------- Load dps / dpp. ----------
    sft, dps_to_use, dpp_to_use = _load_dpp_dps(args, parser, sft)

    # Convert dps to dpp. Easier to manage all the remaining options without
    # multiplying if - else calls.
    if dps_to_use is not None:
        # Then dpp_to_use is None, and the sft contains no dpp key.
        # Can overwrite.
        sft = convert_dps_to_dpp(sft, dps_to_use, overwrite=True)
        all_keys = dps_to_use
    else:
        all_keys = dpp_to_use

    # -------- Format values  ----------
    # In case where we average the dpp, average it now and pretend it's a dps,
    # then re-copy to all dpp.
    if args.mean_streamline or args.mean_endpoints:
        logging.info("Averaging values for all streamlines.")
        for key in all_keys:
            sft.data_per_streamline[key] = perform_operation_dpp_to_dps(
                'mean', sft, key, endpoints_only=args.mean_endpoints)
        sft = convert_dps_to_dpp(sft, all_keys, overwrite=True)

    # -------- Projection and saving ----------
    for key in all_keys:
        logging.info("Projecting streamlines metric {} to a map".format(key))
        the_map = project_dpp_to_map(sft, key, endpoints_only=args.to_endpoints)

        out_file = args.out_prefix + key + '.nii.gz'
        logging.info("Saving file {}".format(out_file))
        nib.save(nib.Nifti1Image(the_map, sft.affine), out_file)


if __name__ == '__main__':
    main()
