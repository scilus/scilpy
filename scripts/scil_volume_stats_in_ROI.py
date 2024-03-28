#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the statistics (mean, std) of scalar maps, which can represent
diffusion metrics, in a ROI. Prints the results.

The mask can either be a binary mask, or a weighting mask. If the mask is
a weighting mask it should either contain floats between 0 and 1 or should be
normalized with --normalize_weights. IMPORTANT: if the mask contains weights
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


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_mask',
                   help='Mask volume filename.\nCan be a binary mask or a '
                        'weighted mask.')

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
        assert_inputs_exist(parser, args.in_mask)

        tmp_file_list = glob.glob(os.path.join(args.metrics_dir, '*nii.gz'))
        args.metrics_file_list = [os.path.join(args.metrics_dir, f)
                                  for f in tmp_file_list]
    else:
        assert_inputs_exist(parser, [args.in_mask] + args.metrics_file_list)
    assert_headers_compatible(parser,
                              [args.in_mask] + args.metrics_file_list)

    # Loading
    mask_data = nib.load(args.in_mask).get_fdata(dtype=np.float32)
    if len(mask_data.shape) > 3:
        parser.error('Mask should be a 3D image.')
    if np.min(mask_data) < 0:
        parser.error('Mask should not contain negative values.')
    roi_name = split_name_with_nii(os.path.basename(args.in_mask))[0]

    # Discussion about the way the normalization is done.
    # https://github.com/scilus/scilpy/pull/202#discussion_r411355609
    # Summary:
    # 1) We don't want to normalize with data = (data-min) / (max-min) because
    # it zeroes out the minimal values of the array. This is not a large error
    # source, but not preferable.
    # 2) data = data / max(data) or data = data / sum(data): in practice, when
    # we use them in numpy using their weights argument, leads to the same
    # result.
    if args.normalize_weights:
        mask_data /= np.max(mask_data)
    elif args.bin:
        mask_data[np.where(mask_data > 0.0)] = 1.0
    elif np.min(mask_data) < 0.0 or np.max(mask_data) > 1.0:
        parser.error('Mask data should only contain values between 0 and 1. '
                     'Try --normalize_weights.')

    # Load and process all metrics files.
    json_stats = {roi_name: {}}
    for f in args.metrics_file_list:
        metric_img = nib.load(f)
        metric_name = split_name_with_nii(os.path.basename(f))[0]
        if len(metric_img.shape) == 3:
            data = metric_img.get_fdata(dtype=np.float64)
            if np.any(np.isnan(data)):
                logging.warning("Metric '{}' contains some NaN. Ignoring "
                                "voxels with NaN."
                                .format(os.path.basename(f)))
            mean, std = weighted_mean_std(mask_data, data)
            json_stats[roi_name][metric_name] = {'mean': mean,
                                                 'std': std}
        else:
            parser.error(
                'Metric {} is not a 3D image ({}D shape).'
                .format(f, len(metric_img.shape)))

    # Print results
    print(json.dumps(json_stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == "__main__":
    main()
