#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes the information from the input metrics for each cortical region
(corresponding to an atlas). If more than one metric are provided, statistics are 
computed separately for each.

Hint: For instance, this script could be useful if you have a seed map from a
specific bundle, to know from which regions it originated.

Formerly: scil_compute_seed_by_labels.py
"""

import argparse
import glob
import json
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels, get_stats_in_label
from scilpy.io.utils import (add_json_args, add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_headers_compatible)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_labels',
                   help='Path of the input label file.')
    p.add_argument('in_labels_lut',
                   help='Path of the LUT file corresponding to labels,'
                        'used to name the regions of interest.')

    g = p.add_argument_group('Metrics input options')
    gg = g.add_mutually_exclusive_group(required=True)
    gg.add_argument('--metrics_dir', metavar='dir',
                    help='Name of the directory containing metrics files: '
                         'we will \nload all nifti files.')
    gg.add_argument('--metrics', dest='metrics_file_list', nargs='+',
                    metavar='file',
                    help='Metrics nifti filename. List of the names of the '
                         'metrics file, \nin nifti format.')
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
        assert_inputs_exist(parser, [args.in_labels, args.in_labels_lut])

        args.metrics_file_list = glob.glob(os.path.join(args.metrics_dir, '*nii.gz'))
    else:
        assert_inputs_exist(parser, [args.in_labels] + args.metrics_file_list)
    assert_headers_compatible(parser,
                              [args.in_labels] + args.metrics_file_list)

    # Loading
    label_data = get_data_as_labels(nib.load(args.in_labels))
    with open(args.in_labels_lut) as f:
        label_dict = json.load(f)

    # Computing stats for all metrics files
    json_stats = {}
    for metric_filename in args.metrics_file_list:
        metric_data = nib.load(metric_filename).get_fdata(dtype=np.float32)
        metric_name = split_name_with_nii(os.path.basename(metric_filename))[0]
        if len(metric_data.shape) > 3:
            parser.error('Input metrics should be 3D images.')

        # Process
        out_dict = get_stats_in_label(metric_data, label_data, label_dict)
        json_stats[metric_name] = out_dict

    if len(args.metrics_file_list) == 1:
        json_stats = json_stats[metric_name]
    print(json.dumps(json_stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == "__main__":
    main()
