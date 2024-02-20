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
"""

import argparse
import logging
import os

import nibabel as nib
from nibabel.streamlines import ArraySequence
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             add_reference_arg,
                             load_matrix_in_any_format, assert_outputs_exist)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.tractanalysis.streamlines_metrics import \
    compute_tract_counts_map


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')
    p.add_argument('out_prefix',
                   help='Folder + prefix to save endpoints metric(s). We will '
                        'save one nifti \nfile per per dpp/dps key given.\n'
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
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _project_metrics(curr_metric_map, count, orig_s, streamline_mean,
                     just_endpoints):
    if just_endpoints:
        xyz = orig_s[0, :].astype(int)
        curr_metric_map[xyz[0], xyz[1], xyz[2]] += streamline_mean
        count[xyz[0], xyz[1], xyz[2]] += 1

        xyz = orig_s[-1, :].astype(int)
        curr_metric_map[xyz[0], xyz[1], xyz[2]] += streamline_mean
        count[xyz[0], xyz[1], xyz[2]] += 1
    else:
        for x, y, z in orig_s[:].astype(int):
            curr_metric_map[x, y, z] += streamline_mean
            count[x, y, z] += 1


def _pick_data(args, sft):
    if args.use_dps or args.load_dps:
        if args.use_dps:
            for dps in args.use_dps:
                if dps not in sft.data_per_streamline:
                    raise IOError('DPS key not in the sft: {}'.format(dps))
            name = args.use_dps
            data = [sft.data_per_streamline[dps] for dps in args.use_dps]
        else:
            name = args.load_dps
            data = [load_matrix_in_any_format(dps) for dps in args.load_dps]
        for i in range(len(data)):
            if len(data[i]) != len(sft):
                raise IOError('DPS length does not match the SFT: {}'
                              .format(name[i]))
    elif args.use_dpp or args.load_dpp:
        if args.use_dpp:
            name = args.use_dpp
            data = [sft.data_per_point[dpp]._data for dpp in args.use_dpp]
        else:
            name = args.load_dpp
            data = [load_matrix_in_any_format(dpp) for dpp in args.load_dpp]
        for i in range(len(data)):
            if len(data[i]) != len(sft.streamlines._data):
                raise IOError('DPP length does not match the SFT: {}'
                              .format(name[i]))
    return zip(name, data)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # -------- General checks ----------
    assert_inputs_exist(parser, [args.in_bundle],
                        args.load_dps + args.load_dpp)

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

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

    if len(sft.streamlines) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return

    for fname, data in _pick_data(args, sft):
        curr_metric_map = np.zeros(sft.dimensions)
        count = np.zeros(sft.dimensions)

        for j in range(len(sft.streamlines)):
            if args.use_dps or args.load_dps:
                streamline_mean = np.mean(data[j])
            else:
                tmp_data = ArraySequence()
                tmp_data._data = data
                tmp_data._offsets = sft.streamlines._offsets
                tmp_data._lengths = sft.streamlines._lengths

                if not args.to_wm:
                    streamline_mean = (np.mean(tmp_data[j][-1])
                                       + np.mean(tmp_data[j][0])) / 2
                else:
                    streamline_mean = np.mean(tmp_data[j])

            _project_metrics(curr_metric_map, count, sft.streamlines[j],
                             streamline_mean, not args.to_wm)

        curr_metric_map[count != 0] /= count[count != 0]
        metric_fname, _ = os.path.splitext(os.path.basename(fname))
        nib.save(nib.Nifti1Image(curr_metric_map, sft.affine),
                 os.path.join(args.out_folder,
                              '{}_endpoints_metric{}'.format(metric_fname,
                                                             '.nii.gz')))


if __name__ == '__main__':
    main()
