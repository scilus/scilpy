#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot mean/std per point.
"""

import argparse
import json
import os

import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.utils.metrics_tools import plot_metrics_stats


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_json',
                   help='JSON file containing the mean/std per point. For '
                        'example, can be created using '
                        'scil_compute_metrics_along_streamline.')
    p.add_argument('out_dir',
                   help='Output directory.')

    p.add_argument('--stats_over_population', action='store_true',
                   help='If set, consider the input stats to be over an '
                        'entire population and not subject-based.')

    p1 = p.add_mutually_exclusive_group()
    p1.add_argument('--fill_color',
                   help='Hexadecimal RGB color filling the region between '
                        'mean +/- std. The hexadecimal RGB color should be '
                        'formatted as 0xRRGGBB.')
    p1.add_argument('--dict_colors',
                    help='Dictionnary mapping basename to color.'
                         'Same convention as --color.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_json)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)

    if args.fill_color and len(args.fill_color) != 8:
        parser.error('Hexadecimal RGB color should be formatted as 0xRRGGBB')

    with open(args.in_json, 'r+') as f:
        if args.stats_over_population:
            mean_std_per_point = json.load(f)
        else:
            mean_std_per_point = list(json.load(f).values())[0]
    
    for bundle_name, bundle_stats in mean_std_per_point.items():
        for metric, metric_stats in bundle_stats.items():
            nb_points = len(metric_stats)
            num_digits_labels = len(str(nb_points))
            means = []
            stds = []
            for label_int in range(1, nb_points+1):
                label = str(label_int).zfill(num_digits_labels)
                mean = metric_stats.get(label, {'mean': np.nan})['mean']
                mean = mean if mean else np.nan
                std = metric_stats.get(label, {'std': np.nan})['std']

                means += [mean]
                stds += [std]

            if args.dict_colors:
                with open(args.dict_colors, 'r') as data:
                    dict_colors = json.load(data)
                color = dict_colors[bundle_name]
            elif args.fill_color is not None:
                color = args.fill_color
            else:
                color = '0x000000'

            fig = plot_metrics_stats(
                np.array(means), np.array(stds),
                title=bundle_name,
                xlabel='Location along the streamline',
                ylabel=metric,
                fill_color=(color.replace("0x", "#")))
            fig.savefig(
                os.path.join(args.out_dir, '{}_{}.png'.format(bundle_name,
                                                              metric)),
                bbox_inches='tight')


if __name__ == '__main__':
    main()
