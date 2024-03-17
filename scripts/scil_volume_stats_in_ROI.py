#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the statistics (mean, std) of scalar maps, which can represent
diffusion metrics, in a ROI.

The mask can either be a binary mask, or a weighting mask. If the mask is
a weighting mask it should either contain floats between 0 and 1 or should be
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
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_headers_compatible)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.metrics_tools import get_roi_metrics_mean_std


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_mask',
                   help='Mask volume filename.\nCan be a binary mask or a '
                        'weighted mask.')

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

    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if args.metrics_dir and os.path.exists(args.metrics_dir):
        list_metrics_files = glob.glob(os.path.join(args.metrics_dir,
                                                    '*nii.gz'))
        assert_inputs_exist(parser, [args.in_mask] + list_metrics_files)
        assert_headers_compatible(parser, [args.in_mask] + list_metrics_files)
    elif args.metrics_file_list:
        assert_inputs_exist(parser, [args.in_mask] + args.metrics_file_list)
        assert_headers_compatible(parser,
                                  [args.in_mask] + args.metrics_file_list)

    # Load mask and validate content depending on flags
    mask_img = nib.load(args.in_mask)

    if len(mask_img.shape) > 3:
        logging.error('Mask should be a 3D image.')

    # Can be a weighted image
    mask_data = mask_img.get_fdata(dtype=np.float32)

    if np.min(mask_data) < 0:
        logging.error('Mask should not contain negative values.')

    # Discussion about the way the normalization is done.
    # https://github.com/scilus/scilpy/pull/202#discussion_r411355609
    if args.normalize_weights:
        mask_data /= np.max(mask_data)

    if np.min(mask_data) < 0.0 or np.max(mask_data) > 1.0:
        parser.error('Mask data should only contain values between 0 and 1. '
                     'Try --normalize_weights.')

    if args.bin:
        mask_data[np.where(mask_data > 0.0)] = 1.0

    # Load all metrics files.
    metrics_files = []
    if args.metrics_dir:
        for f in sorted(os.listdir(args.metrics_dir)):
            metric_img = nib.load(os.path.join(args.metrics_dir, f))
            if len(metric_img.shape)==3:
                # Check if NaNs in metrics
                if np.any(np.isnan(metric_img.get_fdata(dtype=np.float64))):
                    logging.warning('Metric \"{}\" contains some NaN.'.format(metric_img.get_filename()) +
                                    ' Ignoring voxels with NaN.')
                metrics_files.append(metric_img)
            else:
                parser.error('Metric {} is not compatible ({}D image).'.format(os.path.join(args.metrics_dir, f),
                                                                               len(metric_img.shape)))
    elif args.metrics_file_list:
        assert_headers_compatible(parser, [args.in_mask] +
                                  args.metrics_file_list)
        for f in args.metrics_file_list:
            metric_img = nib.load(f)
            if len(metric_img.shape)==3:
                # Check if NaNs in metrics
                if np.any(np.isnan(metric_img.get_fdata(dtype=np.float64))):
                    logging.warning('Metric \"{}\" contains some NaN.'.format(metric_img.get_filename()) +
                                    ' Ignoring voxels with NaN.')
                metrics_files.append(metric_img)
            else:
                parser.error('Metric {} is not compatible ({}D image).'.format(f,
                                                                               len(metric_img.shape)))
    # Compute the mean values and standard deviations
    stats = get_roi_metrics_mean_std(mask_data, metrics_files)

    roi_name = split_name_with_nii(os.path.basename(args.in_mask))[0]
    json_stats = {roi_name: {}}
    for metric_file, (mean, std) in zip(metrics_files, stats):
        metric_name = split_name_with_nii(
            os.path.basename(metric_file.get_filename()))[0]
        json_stats[roi_name][metric_name] = {
            'mean': mean.item(),
            'std': std.item()
        }

    print(json.dumps(json_stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == "__main__":
    main()
