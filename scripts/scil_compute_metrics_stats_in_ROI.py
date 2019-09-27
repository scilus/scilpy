#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute the statistics (mean, stddev) of scalar maps, which can represent
diffusion metrics, in a ROI.

The mask can either be a binary mask, or a weighting mask. If a weighting mask
should either contain floats between 0 and 1, or should be normalized with
--normalize_weights.

IMPORTANT: if the mask contains weights (and not 0 and 1 exclusively), the
standard deviation will also be weighted.
"""

from __future__ import division, print_function

import argparse
import nibabel as nb
import numpy as np
import os
import json

from scilpy.io.utils import assert_inputs_exist
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.metrics_tools import get_metrics_stats_over_volume


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument('mask',
                   help='mask volume file name, formatted in any nibabel ' +
                        'supported format.\nCan be a binary mask or a ' +
                        'weighting mask.')

    mg = p.add_mutually_exclusive_group(required=True)
    mg.add_argument('--metrics_dir',
                    help='metrics files directory. name of the directory ' +
                         'containing the metrics files.')
    mg.add_argument('--metrics',
                    dest='metrics_file_list', nargs='+',
                    help='metrics nifti file name. list of the names of the ' +
                         'metrics file, in nifti format.')

    p.add_argument('--bin',
                   action='store_true',
                   help='if set, will consider every value of the mask ' +
                        'higher than 0 to be part of the mask, and set to 1 ' +
                        '(equivalent weighting for every voxel).')

    p.add_argument('--normalize_weights',
                   action='store_true',
                   help='if set, the weights will be normalized to the [0,1] '
                        'range.')

    p.add_argument('--indent',
                   type=int, default=2,
                   help='Indent for json pretty print.')
    p.add_argument('--sort_keys',
                   action='store_true',
                   help='Sort keys in output json.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.mask)

    # Load mask and validate content depending on flags
    img = nb.load(args.mask)

    if not issubclass(img.get_data_dtype().type, np.floating) and \
            not args.normalize_weights:
        parser.error('The mask file must contain floating point numbers.')

    weighting_data = img.get_data().astype(np.float64)

    if args.normalize_weights:
        weighting_data /= np.sum(weighting_data)

    if np.min(weighting_data) < 0.0 or np.max(weighting_data) > 1.0:
        parser.error('Mask data should only contain values between 0 and 1. '
                     'Try --normalize_weights.')

    if args.bin:
        weighting_data[np.where(weighting_data > 0.0)] = 1.0

    # Load all metrics files, and keep some header information.
    if args.metrics_dir:
        metrics_files = [nb.load(args.metrics_dir + f)
                         for f in sorted(os.listdir(args.metrics_dir))]
    elif args.metrics_file_list:
        metrics_files = [nb.load(f) for f in args.metrics_file_list]

    # Compute the mean values and standard deviations
    stats = get_metrics_stats_over_volume(weighting_data, metrics_files)

    roi_name = split_name_with_nii(os.path.basename(args.mask))[0]
    json_stats = {roi_name: {}}
    for metric_file, (mean, std) in zip(metrics_files, stats):
        metric_name = split_name_with_nii(
            os.path.basename(metric_file.get_filename()))[0]
        json_stats[roi_name][metric_name] = {
            'mean': mean,
            'std': std
        }

    print(json.dumps(json_stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == "__main__":
    main()
