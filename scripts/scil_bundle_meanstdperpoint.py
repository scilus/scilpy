#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.io.image import assert_same_resolution
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_reference_arg,
                             assert_inputs_exist)
from scilpy.tractanalysis import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute mean and standard deviation for all streamlines '
                    'points in the bundle for each metric combination',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file to compute statistics on.')

    p.add_argument('label_map',
                   help='Label map (.npz) of the corresponding '
                        'fiber bundle.')
    p.add_argument('distance_map',
                   help='Distance map (.npz) of the corresponding '
                        'bundle/centroid streamline.')
    p.add_argument('metrics', nargs='+',
                   help='Nifti metric(s) to compute statistics on.')

    p.add_argument('--density_weighting', action='store_true',
                   help='If set, weight statistics by the number of '
                        'streamlines passing through each voxel.')
    p.add_argument('--distance_weighting', action='store_true',
                   help='If set, weight statistics by the inverse of the '
                        'distance between a streamline and the centroid.')

    add_reference_arg(p)
    add_json_args(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bundle, args.label_map,
                                 args.distance_map] + args.metrics)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

    stats = {}
    bundle_name, _ = os.path.splitext(os.path.basename(args.in_bundle))
    if len(sft) == 0:
        stats[bundle_name] = None
        print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))
        return

    assert_same_resolution(args.metrics)
    metrics = [nib.load(metric) for metric in args.metrics]

    if args.density_weighting:
        track_count = compute_tract_counts_map(
            sft.streamlines, metrics[0].shape).astype(np.float64)
    else:
        track_count = np.ones(metrics[0].shape)

    label_file = np.load(args.label_map)
    labels = label_file['arr_0']

    unique_labels = np.unique(labels)
    num_digits_labels = len(str(np.max(unique_labels)))

    distance_file = np.load(args.distance_map)
    distances_to_centroid_streamline = distance_file['arr_0']
    # Bigger weight near the centroid streamline
    distances_to_centroid_streamline = 1.0 / distances_to_centroid_streamline

    if len(labels) != len(distances_to_centroid_streamline):
        raise Exception(
            'Label map doesn\'t contain the same number of '
            'entries as the distance map. {} != {}'
            .format(len(labels), len(distances_to_centroid_streamline)))

    bundle_data_int = sft.streamlines.data.astype(np.int)
    stats[bundle_name] = {}

    for metric in metrics:
        metric_data = metric.get_fdata()
        current_metric_fname, _ = split_name_with_nii(
            os.path.basename(metric.get_filename()))
        stats[bundle_name][current_metric_fname] = {}

        for i in unique_labels:
            number_key = '{}'.format(i).zfill(num_digits_labels)
            label_stats = {}
            stats[bundle_name][current_metric_fname][number_key] = label_stats

            label_indices = bundle_data_int[labels == i]
            label_metric = metric_data[label_indices[:, 0],
                                       label_indices[:, 1],
                                       label_indices[:, 2]]
            track_weight = track_count[label_indices[:, 0],
                                       label_indices[:, 1],
                                       label_indices[:, 2]]
            label_weight = track_weight
            if args.distance_weighting:
                label_weight *= distances_to_centroid_streamline[labels == i]
            if np.sum(label_weight) == 0:
                logging.warning('Weights sum to zero, can\'t be normalized. '
                                'Disabling weighting')
                label_weight = None

            label_mean = np.average(label_metric,
                                    weights=label_weight)
            label_std = np.sqrt(np.average(
                (label_metric - label_mean) ** 2,
                weights=label_weight))
            label_stats['mean'] = float(label_mean)
            label_stats['std'] = float(label_std)

    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
