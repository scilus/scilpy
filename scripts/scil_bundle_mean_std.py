#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute mean and std for each metric.

- Default: For the whole bundle. This is achieved by averaging the metric
  values of all voxels occupied by the bundle.
- Option --per_point: For all streamlines points in the bundle for each metric
  combination, along the bundle, i.e. for each point.
  **To create label_map and distance_map, see
  scil_bundle_label_map.py

Density weighting modifies the contribution of voxel with lower/higher
streamline count to reduce influence of spurious streamlines.

Formerly: scil_compute_bundle_mean_std_per_point.py or
scil_compute_bundle_mean_std.py
"""

import argparse
import json
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.utils.filenames import split_name_with_nii
from scilpy.io.streamlines import (load_tractogram_with_reference,
                                   verify_compatibility_with_reference_sft)
from scilpy.io.utils import (add_json_args,
                             add_reference_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.utils.metrics_tools import get_bundle_metrics_mean_std, \
    get_bundle_metrics_mean_std_per_point


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file to compute statistics on.')
    p.add_argument('in_metrics', nargs='+',
                   help='Nifti file to compute statistics on. Probably some '
                        'tractometry measure(s) such as FA, MD, RD, ...')

    g = p.add_mutually_exclusive_group()
    g.add_argument('--per_point', metavar='in_labels', dest='in_labels',
                   help='If set, computes the metrics per point instead of on '
                        'the whole bundle.\n'
                        'You must then give the label map (.nii.gz) of the '
                        'corresponding fiber bundle.')
    g.add_argument('--include_dps', action='store_true',
                   help='Save values from data_per_streamline.\n'
                        'Currently not offered with option --per_point.')

    p.add_argument('--density_weighting', action='store_true',
                   help='If set, weights statistics by the number of '
                        'fibers passing through each voxel.')
    p.add_argument('--distance_weighting', metavar='DISTANCE_NII',
                   help='If set, weights statistics by the inverse of the '
                        'distance between a streamline and the centroid.')
    p.add_argument('--correlation_weighting', metavar='CORRELATION_NII',
                   help='If set, weight statistics by the correlation '
                        'strength between longitudinal data.')

    p.add_argument('--out_json',
                   help='Path of the output file. If not given, the output '
                        'is simply printed on screen.')
    
    add_reference_arg(p)
    add_json_args(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_bundle] + args.in_metrics,
                        optional=[args.distance_weighting, args.in_labels,
                                  args.correlation_weighting, args.reference])
    assert_outputs_exist(parser, args, '', args.out_json)

    # Load everything and check
    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

    verify_compatibility_with_reference_sft(sft, args.in_metrics,
                                            parser, args)
    metrics = [nib.load(metric) for metric in args.in_metrics]

    bundle_name, _ = os.path.splitext(os.path.basename(args.in_bundle))
    if len(sft) == 0:
        stats = {bundle_name: None}
        print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))
        return

    if args.distance_weighting:
        img = nib.load(args.distance_weighting)
        distances_map = img.get_fdata(dtype=float)
    else:
        distances_map = None

    if args.correlation_weighting:
        img = nib.load(args.correlation_weighting)
        correlation_map = img.get_fdata(dtype=float)
    else:
        correlation_map = None

    for index, metric in enumerate(metrics):
        if np.any(np.isnan(metric.get_fdata())):
            logging.warning('Metric \"{}\" contains some NaN. Ignoring '
                            'voxels with NaN.'.format(args.in_metrics[index]))

    # Now process
    if args.in_labels is None:
        # Whole streamline
        bundle_stats = get_bundle_metrics_mean_std(sft.streamlines,
                                                   metrics,
                                                   distances_map,
                                                   correlation_map,
                                                   args.density_weighting)

        bundle_name, _ = os.path.splitext(os.path.basename(args.in_bundle))

        stats = {bundle_name: {}}
        for metric, (mean, std) in zip(metrics, bundle_stats):
            metric_name = split_name_with_nii(
                os.path.basename(metric.get_filename()))[0]
            stats[bundle_name][metric_name] = {
                'mean': mean,
                'std': std
            }
        if args.include_dps:
            for metric in sft.data_per_streamline.keys():
                mean = float(np.average(sft.data_per_streamline[metric]))
                std = float(np.std(sft.data_per_streamline[metric]))
                stats[bundle_name][metric] = {
                    'mean': mean,
                    'std': std
                }
    else:
        # Per point
        labels_img = nib.load(args.in_labels)
        labels = get_data_as_labels(labels_img)

        stats = get_bundle_metrics_mean_std_per_point(
            sft.streamlines, bundle_name, metrics, labels,
            distances_map, correlation_map, args.density_weighting)

    if args.out_json:
        with open(args.out_json, 'w') as outfile:
            json.dump(stats, outfile,
                      indent=args.indent, sort_keys=args.sort_keys)
    else:
        print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
