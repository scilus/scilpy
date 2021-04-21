#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the statistics (mean, std) of scalar maps, which can represent
diffusion metrics, in a ROI (or multiples ROIs).

ROI mask can either be binary, or a weighted map. If the ROI is
a weighting map it should either contain floats between 0 and 1 or should be
normalized with --normalize_weights.

IMPORTANT: if the mask contains weights (and not 0 and 1 exclusively), the
standard deviation will also be weighted.
"""

import argparse
import glob
import json
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_json_args,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.metrics_tools import get_roi_metrics_mean_std


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_mask', nargs='+',
                   help='Mask volume filename (ROI).\nCan be a binary mask '
                        'or a weighted mask.')

    p_metric = p.add_argument_group('Metrics input options')
    g_metric = p_metric.add_mutually_exclusive_group(required=True)
    g_metric.add_argument('--metrics_dir',
                          help='Metrics files directory. Name of the '
                               'directory containing the metrics files.')
    g_metric.add_argument('--metrics', dest='metrics_file_list', nargs='+',
                          help='Metrics nifti filename. List of the names of '
                               'the metrics file, in nifti format.')

    p.add_argument('--bin', action='store_true',
                   help='If set, will consider every value of the mask '
                        'higher than 0 to be part of the mask, and set to 1 '
                        '(equivalent weighting for every voxel).')

    p.add_argument('--normalize_weights', action='store_true',
                   help='If set, the weights will be normalized to the [0,1] '
                        'range.')

    p.add_argument('--out_file',
                   help='Save all average to a file (txt, json).')

    add_overwrite_arg(p)
    add_json_args(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.metrics_dir and os.path.exists(args.metrics_dir):
        list_metrics_files = glob.glob(os.path.join(args.metrics_dir,
                                                    '*nii.gz'))
        assert_inputs_exist(parser, args.in_mask + list_metrics_files)
    elif args.metrics_file_list:
        assert_inputs_exist(parser, args.in_mask + args.metrics_file_list)

    assert_outputs_exist(parser, args, [], optional=args.out_file)

    # Load masks and validate content depending on flags
    mask_list = []
    for mask_file in args.in_mask:
        mask_img = nib.load(mask_file)

        if len(mask_img.shape) > 3:
            logging.error('Masks should be a 3D image.')

        # Can be a weighted image
        mask_data = mask_img.get_fdata()

        if np.min(mask_data) < 0:
            logging.error('Masks should not contain negative values.')

        # Discussion about the way the normalization is done.
        # https://github.com/scilus/scilpy/pull/202#discussion_r411355609
        if args.normalize_weights:
            mask_data /= np.max(mask_data)

        if np.min(mask_data) < 0.0 or np.max(mask_data) > 1.0:
            parser.error('Masks should only contain values between 0 and 1. '
                         'Try --normalize_weights.')

        if args.bin:
            mask_data[np.where(mask_data > 0.0)] = 1.0

        mask_list.append(mask_data)

    # Load all metrics files.
    if args.metrics_dir:
        metrics_files = [nib.load(args.metrics_dir + f)
                         for f in sorted(os.listdir(args.metrics_dir))]
    elif args.metrics_file_list:
        metrics_files = [nib.load(f) for f in args.metrics_file_list]

    # Compute the mean values and standard deviations
    json_stats = {}
    avg_array = np.zeros([len(mask_list), len(metrics_files)], dtype=float)
    for i, mask_file in enumerate(args.in_mask):
        roi_name = split_name_with_nii(os.path.basename(mask_file))[0]
        stats = get_roi_metrics_mean_std(mask_list[i], metrics_files)
        json_stats[roi_name] = {}

        for j, (metric_file, (mean, std)) \
                in enumerate(zip(metrics_files, stats)):
            metric_name = split_name_with_nii(
                os.path.basename(metric_file.get_filename()))[0]
            json_stats[roi_name][metric_name] = {
                'mean': mean.item(),
                'std': std.item()
            }
            avg_array[i, j] = mean.item()

    print(json.dumps(json_stats, indent=args.indent, sort_keys=args.sort_keys))

    if args.out_file:
        _, file_ext = os.path.splitext(args.out_file)

        if file_ext == ".json":
            with open(args.out_file, 'w') as fp:
                json.dump(json_stats, fp, indent=args.indent,
                          sort_keys=args.sort_keys)
        else:
            np.savetxt(args.out_file, avg_array, delimiter=",", newline=";")


if __name__ == "__main__":
    main()
