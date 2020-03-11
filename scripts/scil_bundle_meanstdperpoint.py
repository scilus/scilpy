#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute mean and standard deviation for all streamlines points in the bundle
for each metric combination, along the bundle, i.e. for each point.
"""

import argparse
import json
import os

import nibabel as nib
import numpy as np

from scilpy.io.image import assert_same_resolution
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_reference_arg,
                             assert_inputs_exist)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.metrics_tools import get_bundle_metrics_meanstdperpoint


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
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
                   help='Nifti file to compute statistics on. Probably some '
                        'tractometry measure(s) such as FA, MD, RD, ...')

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

    bundle_name, _ = os.path.splitext(os.path.basename(args.in_bundle))
    if len(sft) == 0:
        stats = {bundle_name: None}
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

    distance_file = np.load(args.distance_map)
    distances_to_centroid_streamline = distance_file['arr_0']

    if len(labels) != len(distances_to_centroid_streamline):
        raise Exception(
            'Label map doesn\'t contain the same number of '
            'entries as the distance map. {} != {}'
            .format(len(labels), len(distances_to_centroid_streamline)))

    stats = get_bundle_metrics_meanstdperpoint(sft.streamlines, bundle_name,
                                               distances_to_centroid_streamline,
                                               metrics, track_count, labels,
                                               args.distance_weighting)

    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
