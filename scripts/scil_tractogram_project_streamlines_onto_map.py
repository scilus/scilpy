#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projects metrics onto the underlying voxels of a streamlines.

How to load the metrics for each point of the streamline:
    - From metric maps: uses the value of underlying voxels.
    - From dps: uses the same value for each point of the streamline.
    - From dpp: one value per point.

How to use the metrics:
    1. Average all points of the streamline to get a mean value.
    2. Average the two endpoints and get their mean value.
       (Not possible with dps)
    3. Keep each point individually.
       (Not possible with dps)

How to project values to a map:
    A. Project the result to each point of the streamline.
    B. Project the result to the enpoints only.

When possible, streamlines will be uncompressed, i.e. we will find all voxels
touching a segment, even if no point of the streamline lies in it.

The 6 possible combinations are:
    -1A. Projects the mean value of each streamline to its underlying voxels.
         (Uses uncompressed)
    -1B. Projects the mean value of each streamline to the endpoint voxels.
    -2A. Projects the mean enpoint value to the whole extend of the streamline.
         (Uses uncompressed)
    -2B. Projects the mean endpoint value to the endpoint voxels.
    -3A. Projects each point to its underlying voxel.
         (Can uncompress with data from in_metrics, but can't uncompress with
         data from dpp; we wouldn't know how to uncompress the dpp values.)
    -3B. Projects the two endpoint values to their underlying voxels.

In voxels containing many streamlines, values of all streamlines will be
averaged together.
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np
from scilpy.tractograms.uncompress import uncompress

from scilpy.io.streamlines import (load_tractogram_with_reference,
                                   load_dps_files_as_dps,
                                   load_dpp_files_as_dpp,
                                   load_map_values_as_dpp,
                                   verify_compatibility_with_reference_sft)
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)

from scilpy.tractograms.dps_and_dpp_managememt import (
    average_dpp_as_dps, project_dpp_to_map, keep_only_endpoints,
    repeat_dps_as_dpp)
from scilpy.utils.filenames import split_name_with_nii


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')
    p.add_argument('out_prefix',
                   help='Folder + prefix to save endpoints metric(s). We will '
                        'save one nifti \nfile per in_metrics, or one per '
                        'dpp/dps key given. \n'
                        'Ex: my_path/subjX_bundleY_ with --use_dpp key1 '
                        'will output \nmy_path/subjX_bundleY_key1.nii.gz')

    p1 = p.add_argument_group(
        description='Where to get the statistics from. (Choose one)')
    p1 = p1.add_mutually_exclusive_group(required=True)
    p1.add_argument('--in_metrics', nargs='+', default=[],
                    help='Nifti metric(s) to compute statistics on. Projects '
                         'the value \nof underlying voxels onto the '
                         'streamlines, loading them as dpp.')
    p1.add_argument('--use_dps', metavar='key', nargs='+',
                    help='Use the data_per_streamline from the tractogram.\n'
                         'It must be a trk.')
    p1.add_argument('--use_dpp', metavar='key', nargs='+', default=[],
                    help='Use the data_per_point from the tractogram. \n'
                         'It must be a trk.')
    p1.add_argument('--load_dps', metavar='file', nargs='+', default=[],
                    help='Load data per streamline (scalar) .txt or .npy.\n'
                         'Must load an array with the right shape.')
    p1.add_argument('--load_dpp', metavar='file', nargs='+', default=[],
                    help='Load data per point (scalar) from .txt or .npy.\n'
                         'Must load an array with the right shape.')

    p2 = p.add_argument_group(description='Processing choice')
    p2 = p2.add_mutually_exclusive_group(required=True)
    p2.add_argument('--mean_endpoints', action='store_true',
                    help="Uses one single value per streamline: the mean "
                         "of the two endpoints.")
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
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # -------- General checks ----------
    if (args.use_dps or args.load_dps) and not args.mean_streamline:
        parser.error("data_per_streamline (dps) usage is only compatiable "
                     "with the --mean_streamline processing choice.")

    assert_inputs_exist(parser, [args.in_bundle],
                        args.in_metrics + args.load_dps + args.load_dpp)

    # Find all final output files
    if args.in_metrics or args.load_dps or args.load_dpp:
        files = args.in_metrics or args.load_dps or args.load_dpp
        metrics_names = []
        for file in files:
            # Prepare dpp key from filename.
            name = os.path.basename(file)
            name, ext = split_name_with_nii(name)
            metrics_names.append(name)
    else:
        metrics_names = args.use_dpp or args.use_dps
    out_files = [args.out_prefix + m for m in metrics_names]
    assert_outputs_exist(parser, args, out_files)

    # -------- Loading streamlines and checking compatibility ----------
    logging.info("Loading tractogram {}".format(args.in_bundle))
    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

    if args.in_metrics:
        verify_compatibility_with_reference_sft(sft, args.in_metrics,
                                                parser, args)

    if len(sft.streamlines) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return

    # -------- Loading dps / dpp. ----------

    # In call cases: only one of the values below can be set at the time.
    dps_to_use = None
    dpp_to_use = None

    # 1. With options --use_dps, --use_dpp: check that dps / dpp key is found.
    # 2. With --load_dps, --load_dpp: Load them now to SFT.
    # 3. With option --in_metrics: format streamlines and load as dpp.
    if args.use_dps:
        dps_to_use = args.use_dps
        for key in args.use_dps:
            if key not in sft.data_per_streamline:
                parser.error('DPS key not in the sft: {}'.format(key))
        logging.info("data_per_streamline key(s) correctly found.")
    elif args.use_dpp:
        dpp_to_use = args.use_dpp
        for key in args.use_dpp:
            if key not in sft.data_per_point:
                parser.error('DPP key not in the sft: {}'.format(key))
            logging.info("data_per_point key(s) correctly found.")
    elif args.load_dps:
        logging.info("Loading dps from file.")
        sft, dps_to_use = load_dps_files_as_dps(parser, args.load_dps, sft)
    elif args.load_dpp:
        # Loading dpp for all points even if we won't use them all to make
        # sure that the loaded files have the correct shape.
        logging.info("Loading dpp from file")
        sft, dpp_to_use = load_dpp_files_as_dpp(parser, args.load_dpp, sft)
    # else, args.in_metrics, but we will refactor the streamlines first.

    # Verify that we have singular values. (Ex, not colors)
    # Remove unused keys

    # Need to list keys before; changes during iteration.
    all_keys = list(sft.data_per_point.keys())
    for key in all_keys:
        if dpp_to_use is not None and key in dpp_to_use:
            if len(sft.data_per_point[key][0].squeeze().shape) > 1:
                raise ValueError(
                    "Expecting scalar values as data_per_point.  Got data of "
                    "shape {}.".format(sft.data_per_point[key][0].shape[1]))
        else:
            del sft.data_per_point[key]

    all_keys = list(sft.data_per_streamline.keys())
    for key in all_keys:
        if dps_to_use is not None and key in dps_to_use:
            if not np.array_equal(
                    sft.data_per_streamline[key][0].squeeze().shape, [1,]):
                raise ValueError(
                    "Expecting scalar values as data_per_streamline. Got data "
                    "of shape {}."
                    .format(sft.data_per_streamline[key][0].shape))
        else:
            del sft.data_per_streamline[key]

    # Ok, now ready for last case. Returns None for other points if we only
    # need the endpoint to avoid non-useful interpolation.
    if args.in_metrics:
        sft, dpp_to_use = load_map_values_as_dpp(
            sft, args.in_metrics, metrics_names, uncompress_first=True,
            endpoints_only=(args.mean_endpoints or args.to_endpoints))

    # -------- Formatting streamlines  ----------

    # In case where we average the dpp, average it now and pretend it's a dps.
    if (args.mean_streamline or args.mean_endpoints) and \
            dpp_to_use is not None:
        sft = average_dpp_as_dps(sft, dpp_to_use, remove_dpp=True,
                                 endpoints_only=args.mean_endpoints)
        dps_to_use = dpp_to_use
        dpp_to_use = None

    # Uncompress if necessary. Already done with --in_metrics.
    if not (args.in_metrics or (args.to_wm and args.point_by_point)):
        logging.info("Uncompressing streamlines...")
        sft.streamlines = uncompress(sft.streamlines)

    # Now, if we we project to_endpoints, keep only the endpoint coordinates.
    if args.to_endpoints:
        sft = keep_only_endpoints(sft)

    # Finally, if we are dealing with dps, convert to dpp (copy the same
    # value everywhere).
    if dps_to_use is not None:
        assert dpp_to_use is None
        sft = repeat_dps_as_dpp(sft, dps_to_use, remove_dps=True)
        dpp_to_use = dps_to_use
    else:
        assert dpp_to_use is not None

    # -------- Projection  ----------

    maps = project_dpp_to_map(sft, dpp_to_use)

    # -------- Save final maps ----------
    for i, the_map in enumerate(maps):
        print("Saving ", out_files[i])
        nib.save(nib.Nifti1Image(the_map, sft.affine), out_files[i])


if __name__ == '__main__':
    main()
