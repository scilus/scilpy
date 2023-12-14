#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projects metrics onto the endpoints of streamlines. The idea is to visualize
the cortical areas affected by metrics (assuming streamlines start/end in
the cortex).

This script can project data from maps (--in_metrics), from data_per_point
(dpp) or data_per_streamline (dps): --load_dpp and --load_dps require an array
from a file (must be the right shape), --use_dpp and --use_dps work only for
.trk file and the key must exist in the metadata.

The default options will take data from endpoints and project it to endpoints.
--from_wm will use data from whole streamlines.
--to_wm will project the data to whole streamline coverage.
This creates 4 combinations of data source and projection.
"""

import argparse
import logging
import os

import nibabel as nib
from nibabel.streamlines import ArraySequence
import numpy as np

from scilpy.io.image import assert_same_resolution
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             add_reference_arg,
                             load_matrix_in_any_format)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.tractanalysis.streamlines_metrics import \
    compute_tract_counts_map
from scilpy.tractograms.uncompress import uncompress


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')
    p.add_argument('out_folder',
                   help='Folder where to save endpoints metric.')

    p1 = p.add_mutually_exclusive_group(required=True)
    p1.add_argument('--in_metrics', nargs='+', default=[],
                    help='Nifti metric(s) to compute statistics on.')
    p1.add_argument('--use_dps', metavar='DPS_KEY', nargs='+',
                    help='Use the data_per_streamline (scalar) from file, '
                         'e.g. commit_weights.')
    p1.add_argument('--use_dpp', metavar='DPP_KEY', nargs='+', default=[],
                    help='Use the data_per_point (scalar) from file.')
    p1.add_argument('--load_dps', metavar='DPS_KEY', nargs='+', default=[],
                    help='Load data per streamline (scalar) .txt or .npy.')
    p1.add_argument('--load_dpp', metavar='DPP_KEY', nargs='+', default=[],
                    help='Load data per point (scalar) from .txt or .npy.')

    p.add_argument('--from_wm', action='store_true',
                   help='Project metrics from whole streamlines coverage.')
    p.add_argument('--to_wm', action='store_true',
                   help='Project metrics into streamlines coverage.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _compute_streamline_mean(cur_ind, cur_min, cur_max, data):
    # From the precomputed indices, compute the binary map
    # and use it to weight the metric data for this specific streamline.
    cur_range = tuple(cur_max - cur_min)

    if len(cur_ind) == 2:
        streamline_density = np.zeros(cur_range, dtype=int)
        streamline_density[cur_ind[:, 0], cur_ind[:, 1]] = 1
    else:
        streamline_density = compute_tract_counts_map(ArraySequence([cur_ind]),
                                                      cur_range)
    streamline_data = data[cur_min[0]:cur_max[0],
                           cur_min[1]:cur_max[1],
                           cur_min[2]:cur_max[2]]
    streamline_average = np.average(streamline_data,
                                    weights=streamline_density)
    return streamline_average


def _process_streamlines(streamlines, just_endpoints):
    # Compute the bounding boxes and indices for all streamlines.
    # just_endpoints will get the indices of the endpoints only for the
    # usecase of projecting GM metrics into the WM.
    mins = []
    maxs = []
    offset_streamlines = []

    # Offset the streamlines to compute the indices only in the bounding box.
    # Reduces memory use later on.
    for idx, s in enumerate(streamlines):
        mins.append(np.min(s.astype(int), 0))
        maxs.append(np.max(s.astype(int), 0) + 1)
        if just_endpoints:
            s = np.stack((s[0, :], s[-1, :]), axis=0)
        offset_streamlines.append((s - mins[-1]).astype(np.float32))

    offset_streamlines = ArraySequence(offset_streamlines)

    if not just_endpoints:
        indices = uncompress(offset_streamlines)
    else:
        indices = ArraySequence()
        indices._offsets = offset_streamlines._offsets
        indices._lengths = offset_streamlines._lengths
        indices._data = np.floor(offset_streamlines._data).astype(int)

    return mins, maxs, indices


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

    assert_inputs_exist(parser, [args.in_bundle], args.in_metrics +
                        args.load_dps + args.load_dpp)
    assert_output_dirs_exist_and_empty(parser, args,
                                       args.out_folder,
                                       create_dir=True)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

    if len(sft.streamlines) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return

    mins, maxs, indices = _process_streamlines(sft.streamlines,
                                               not args.from_wm)

    if args.in_metrics:
        assert_same_resolution(args.in_metrics)
        metrics = [nib.load(metric) for metric in args.in_metrics]
        for metric in metrics:
            data = metric.get_fdata(dtype=np.float32)
            curr_metric_map = np.zeros(metric.shape)
            count = np.zeros(metric.shape)
            for cur_min, cur_max, cur_ind, orig_s in zip(mins, maxs, indices,
                                                         sft.streamlines):
                streamline_mean = _compute_streamline_mean(cur_ind,
                                                           cur_min,
                                                           cur_max,
                                                           data)

                _project_metrics(curr_metric_map, count, orig_s,
                                 streamline_mean, not args.to_wm)
            curr_metric_map[count != 0] /= count[count != 0]
            metric_fname, ext = split_name_with_nii(
                os.path.basename(metric.get_filename()))
            nib.save(nib.Nifti1Image(curr_metric_map, metric.affine,
                                     metric.header),
                     os.path.join(args.out_folder,
                                  '{}_endpoints_metric{}'.format(metric_fname,
                                                                 ext)))
    else:
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
