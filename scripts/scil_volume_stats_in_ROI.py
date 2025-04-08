#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the statistics (mean, std) of scalar maps, which can represent
diffusion metrics, in ROIs. Prints the results.

The ROIs can either be binary masks, or weighting masks. If the ROIs are
 weighting masks, they should either contain floats between 0 and 1 or should be
normalized with --normalize_weights. IMPORTANT: if the ROIs contain weights
(and not 0 and 1 exclusively), the standard deviation will also be weighted.
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
from scilpy.utils.metrics_tools import weighted_mean_std
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_rois', nargs='+',
                   help='ROIs volume filenames.\nCan be binary masks or '
                        'weighted masks.')

    g = p.add_argument_group('Metrics input options')
    gg = g.add_mutually_exclusive_group(required=True)
    gg.add_argument('--metrics_dir', metavar='dir',
                    help='Name of the directory containing metrics files: '
                         'we will \nload all nifti files.')
    gg.add_argument('--metrics', dest='metrics_file_list', nargs='+',
                    metavar='file',
                    help='Metrics nifti filename. List of the names of the '
                         'metrics file, \nin nifti format.')

    p.add_argument('--bin', action='store_true',
                   help='If set, will consider every value of the mask higher'
                        'than 0 to be \npart of the mask (equivalent '
                        'weighting for every voxel).')
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

    # Verifications

    # Get input list. Either all files in dir or given list.
    if args.metrics_dir:
        if not os.path.exists(args.metrics_dir):
            parser.error("Metrics directory does not exist: {}"
                         .format(args.metrics_dir))
        assert_inputs_exist(parser, args.in_rois)

        args.metrics_file_list = glob.glob(os.path.join(args.metrics_dir, '*nii.gz'))
    else:
        assert_inputs_exist(parser, args.in_rois + args.metrics_file_list)
    assert_headers_compatible(parser, args.in_rois + args.metrics_file_list)

    # Computing stats for all ROIs and metrics files
    json_stats = {}
    for roi_filename in args.in_rois:
        roi_data = nib.load(roi_filename).get_fdata(dtype=np.float32)
        if len(roi_data.shape) > 3:
            parser.error('ROI {} should be a 3D image.'.format(roi_filename))
        if np.min(roi_data) < 0:
            parser.error('ROI {} should not contain negative values.'
                         .format(roi_filename))
        roi_name = split_name_with_nii(os.path.basename(roi_filename))[0]

        # Discussion about the way the normalization is done.
        # https://github.com/scilus/scilpy/pull/202#discussion_r411355609
        # Summary:
        # 1) We don't want to normalize with data = (data-min) / (max-min)
        # because it zeroes out the minimal values of the array. This is
        # not a large error source, but not preferable.
        # 2) data = data / max(data) or data = data / sum(data): in practice,
        # when we use them in numpy using their weights argument, leads to the
        # same result.
        if args.normalize_weights:
            roi_data /= np.max(roi_data)
        elif args.bin:
            roi_data[np.where(roi_data > 0.0)] = 1.0
        elif np.min(roi_data) < 0.0 or np.max(roi_data) > 1.0:
            parser.error('ROI {} data should only contain values between 0 '
                         'and 1. Try --normalize_weights.'
                         .format(roi_filename))

        # Load and process all metrics files.
        json_stats[roi_name] = {}
        for f in args.metrics_file_list:
            metric_img = nib.load(f)
            metric_name = split_name_with_nii(os.path.basename(f))[0]
            if len(metric_img.shape) == 3:
                data = metric_img.get_fdata(dtype=np.float64)
                if np.any(np.isnan(data)):
                    logging.warning("Metric '{}' contains some NaN. Ignoring "
                                    "voxels with NaN."
                                    .format(os.path.basename(f)))

                if not roi_data.any():
                    logging.warning("ROI '{}' is empty. "
                                    "Putting NaN as mean and std."
                                    .format(roi_name))
                    mean = np.nan
                    std = np.nan
                else:
                    mean, std = weighted_mean_std(roi_data, data)

                json_stats[roi_name][metric_name] = {'mean': mean,
                                                     'std': std}
            else:
                parser.error(
                    'Metric {} is not a 3D image ({}D shape).'
                    .format(f, len(metric_img.shape)))

    if len(args.in_rois) == 1:
        json_stats = json_stats[roi_name]

    # Print results
    print(json.dumps(json_stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == "__main__":
    main()
