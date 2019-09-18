#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import os

import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_dir_exists_and_empty)
from scilpy.utils.metrics_tools import plot_metrics_stats


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Plot mean/std per point',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('meanstdperpoint',
                   help='JSON file containing the mean/std per point.')
    p.add_argument('output',
                   help='Output directory.')
    p.add_argument('--fill_color',
                   help='Hexadecimal RGB color filling the region between '
                        'mean ± std. The hexadecimal RGB color should be '
                        'formatted as 0xRRGGBB.')
    p.add_argument('--nb_points',
                   type=int, default=20,
                   help='Number of points defining the centroid streamline.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.meanstdperpoint)
    assert_outputs_dir_exists_and_empty(parser, args, args.output)

    if args.fill_color and len(args.fill_color) != 8:
        parser.error('Hexadecimal RGB color should be formatted as 0xRRGGBB')

    with open(args.meanstdperpoint, 'r+') as f:
        meanstdperpoint = json.load(f)

    for bundle_name, bundle_stats in meanstdperpoint.iteritems():
        for metric, metric_stats in bundle_stats.iteritems():
            num_digits_labels = len(str(args.nb_points))
            means = []
            stds = []
            for label_int in xrange(1, args.nb_points+1):
                label = str(label_int).zfill(num_digits_labels)
                mean = metric_stats.get(label, {'mean': np.nan})['mean']
                mean = mean if mean else np.nan
                std = metric_stats.get(label, {'std': np.nan})['std']
                std = std if std else np.nan
                means += [mean]
                stds += [std]

            fig = plot_metrics_stats(
                np.array(means), np.array(stds),
                title=bundle_name,
                xlabel='Location along the streamline',
                ylabel=metric,
                fill_color=(args.fill_color.replace("0x", "#")
                            if args.fill_color else None))
            fig.savefig(
                os.path.join(args.output,
                             '{}_{}.png'.format(bundle_name, metric)),
                bbox_inches='tight')


if __name__ == '__main__':
    main()
