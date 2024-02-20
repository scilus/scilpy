#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot all mean/std per point for a subject or population json file from
tractometry-flow.
WARNING: For population, the displayed STDs is only showing the variation
of the means. It does not account intra-subject STDs.

Formerly: scil_plot_mean_std_per_point.py
"""

import argparse
import itertools
import json
import logging
import os

import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             add_verbose_arg)
from scilpy.utils.metrics_tools import plot_metrics_stats


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_json',
                   help='JSON file containing the mean/std per point. For '
                        'example, can be created using '
                        'scil_bundle_mean_std.py.')
    p.add_argument('out_dir',
                   help='Output directory.')

    p.add_argument('--stats_over_population', action='store_true',
                   help='If set, consider the input stats to be over an '
                        'entire population and not subject-based.')
    p.add_argument('--nb_pts', type=int,
                   help='Force the number of divisions for the bundles.\n'
                        'Avoid unequal plots across datasets, replace missing '
                        'data with zeros.')
    p.add_argument('--display_means', action='store_true',
                   help='Display the subjects means as semi-transparent line.'
                        '\nPoor results when the number of subject is high.')

    p1 = p.add_mutually_exclusive_group()
    p1.add_argument('--fill_color',
                    help='Hexadecimal RGB color filling the region between '
                    'mean +/- std. The hexadecimal RGB color should be '
                    'formatted as 0xRRGGBB.')
    p1.add_argument('--dict_colors',
                    help='Dictionnary mapping basename to color.'
                         'Same convention as --color.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

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
            nb_points = args.nb_pts if args.nb_pts is not None \
                else len(metric_stats)
            num_digits_labels = len(list(metric_stats.keys())[0])
            means = []
            stds = []
            for label_int in range(1, nb_points+1):
                label = str(label_int).zfill(num_digits_labels)
                mean = metric_stats.get(label, {'mean': 0})['mean']
                std = metric_stats.get(label, {'std': 0})['std']
                if not isinstance(mean, list):
                    mean = [mean]
                    std = [std]

                means += [mean]
                stds += [std]

            color = None
            if args.dict_colors:
                with open(args.dict_colors, 'r') as data:
                    dict_colors = json.load(data)
                # Supports variation from rbx-flow
                for key in dict_colors.keys():
                    if key in bundle_name:
                        color = dict_colors[key]
            elif args.fill_color is not None:
                color = args.fill_color
            if color is None:
                color = '0x000000'

            # Robustify for missing data
            means = np.array(list(itertools.zip_longest(*means,
                                                        fillvalue=np.nan))).T
            stds = np.array(list(itertools.zip_longest(*stds,
                                                       fillvalue=np.nan))).T
            for i in range(len(means)):
                _nan = np.isnan(means[i, :])
                if np.count_nonzero(_nan) > 0:
                    if np.count_nonzero(_nan) < len(means[i, :]):
                        means[i, _nan] = np.average(means[i, ~_nan])
                        stds[i, _nan] = np.average(stds[i, ~_nan])
                    else:
                        means[i, _nan] = -1
                        stds[i, _nan] = -1
            if not args.stats_over_population:
                means = np.squeeze(means)
                stds = np.squeeze(stds)
            fig = plot_metrics_stats(means, stds,
                                     title=bundle_name,
                                     xlabel='Location along the streamline',
                                     ylabel=metric,
                                     fill_color=(color.replace("0x", "#")),
                                     display_means=args.display_means)
            fig.savefig(os.path.join(args.out_dir,
                                     '{}_{}.png'.format(bundle_name,
                                                        metric)),
                        bbox_inches='tight')


if __name__ == '__main__':
    main()
