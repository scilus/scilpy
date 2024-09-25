#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to display a histogram of a metric (FA, MD, etc.) from a binary mask
(wm mask, vascular mask, ect.).
These two images must be coregister with each other.

>>> scil_viz_volume_histogram.py metric.nii.gz mask_bin.nii.gz 8
    out_filename_image.png
"""

import argparse
import logging

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg,
                             assert_headers_compatible)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_metric',
                   help='Metric map ex : FA, MD,... .')
    p.add_argument('in_mask',
                   help='Binary mask data to extract value.')
    p.add_argument('n_bins', type=int,
                   help='Number of bins to use for the histogram.')
    p.add_argument('out_png',
                   help='Output filename for the figure.')

    hist = p.add_argument_group(title='Histogram options')
    hist.add_argument('--title', default='Histogram',
                      help='Use the provided info for the histogram title.'
                           ' [%(default)s]')
    hist.add_argument('--x_label',  default='x',
                      help='Use the provided info for the x axis name.')
    hist.add_argument('--colors', default='#0504aa',
                      help='Use the provided info for the bars color.'
                           ' [%(default)s]')

    p.add_argument('--show_only', action='store_true',
                   help='Do not save the figure, only display.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_metric, args.in_mask])
    assert_outputs_exist(parser, args, args.out_png)
    assert_headers_compatible(parser, [args.in_metric, args.in_mask])

    # Load metric image
    metric_img = nib.load(args.in_metric)
    metric_img_data = metric_img.get_fdata(dtype=np.float32)

    # Load mask image
    mask = get_data_as_mask(nib.load(args.in_mask))

    # Select value from mask
    curr_data = metric_img_data[np.where(mask > 0)]

    # Display figure
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(curr_data, bins=args.n_bins,
                               color=args.colors, alpha=0.5, rwidth=0.85)
    plt.xlabel(args.x_label)
    plt.title(args.title)

    if args.show_only:
        plt.show()
    else:
        plt.savefig(args.out_png, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
