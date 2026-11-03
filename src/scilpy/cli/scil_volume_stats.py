#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes some statistics on your volume (maximum, mininimum, mean, etc)
and prints them in the terminal.

See also:
    scil_header_print_info
    scil_volume_count_non_zero_voxels
    scil_volume_stats_in_labels
    scil_volume_stats_in_ROI
"""

import argparse
import logging
import os

import json
import nibabel as nib
import numpy as np

from scilpy.image.volume_operations import count_non_zero_voxels
from scilpy.io.utils import add_json_args, add_verbose_arg, \
    assert_inputs_exist
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_image', metavar='IN_FILE',
                   help='Input file name, in nifti format.')
    p.add_argument('--precision', type=int, default=4,
                   help="Precision level for float display. Default: 4")

    add_verbose_arg(p)
    add_json_args(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image)

    # Load image file
    im = nib.load(args.in_image).get_fdata(dtype=np.float32)
    im = im.flatten()

    # General stats
    nb_voxels = len(im)
    max = np.max(im)
    min = np.min(im)
    mean = np.mean(im)

    # Stats in non-zero voxels
    nb_non_zero = count_non_zero_voxels(im)
    im = im[im!=0]
    max2 = np.max(im)
    min2 = np.min(im)
    mean2 = np.mean(im)

    res = {
        'Amongst all voxels:': {
            'Maximum': float(max),
            'Minimum': float(min),
            'Mean': float(mean),
            'Number of voxels': nb_voxels
            },
        'Amongst non-zero voxels:': {
            'Maximum': float(max2),
            'Minimum': float(min2),
            'Mean': float(mean2),
            'Number of voxels': nb_non_zero
            }
        }

    print(json.dumps(res, indent=args.indent))

if __name__ == "__main__":
    main()
