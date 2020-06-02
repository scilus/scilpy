#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute mean and standard deviation for all streamlines points in the bundle
for each metric combination, along the bundle, i.e. for each point.

**To create label_map and distance_map, see scil_label_and_distance_maps.py.
"""

import argparse
import json
import os

import nibabel as nib
import numpy as np

from scilpy.io.image import assert_same_resolution
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args, add_reference_arg,
                             add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.utils.metrics_tools import get_bundle_metrics_mean_std_per_point


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file to compute statistics on.')

    p.add_argument('label_map',
                   help='Label map (.npz) of the corresponding fiber bundle.')
    p.add_argument('distance_map',
                   help='Distance map (.npz) of the corresponding bundle/'
                        'centroid streamline.')
    p.add_argument('metrics', nargs='+',
                   help='Nifti file to compute statistics on. Probably some '
                        'tractometry measure(s) such as FA, MD, RD, ...')

    p.add_argument('--density_weighting', action='store_true',
                   help='If set, weight statistics by the number of '
                        'streamlines passing through each voxel.')
    p.add_argument('--distance_weighting', action='store_true',
                   help='If set, weight statistics by the inverse of the '
                        'distance between a streamline and the centroid.')
    p.add_argument('--out_json',
                   help='Path of the output json file. If not given, json '
                        'formatted stats are simply printed.')

    add_overwrite_arg(p)
    add_reference_arg(p)
    add_json_args(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bundle, args.label_map,
                                 args.distance_map] + args.metrics)
    assert_outputs_exist(parser, args, '', args.out_json)

    # Load everything
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

    label_file = np.load(args.label_map)
    labels = label_file['arr_0']

    distance_file = np.load(args.distance_map)
    distances_to_centroid_streamline = distance_file['arr_0']

    if len(labels) != len(distances_to_centroid_streamline):
        raise Exception(
            "Label map doesn't contain the same number of entries as the "
            "distance map. {} != {}".format(len(labels),
                                            len(distances_to_centroid_streamline)))

    # Compute stats
    stats = get_bundle_metrics_mean_std_per_point(sft.streamlines, bundle_name,
                                                  distances_to_centroid_streamline,
                                                  metrics, labels,
                                                  args.density_weighting,
                                                  args.distance_weighting)

    if args.out_json:
        with open(args.out_json, 'w') as outfile:
            json.dump(stats, outfile, indent=args.indent,
                      sort_keys=args.sort_keys)
    else:
        print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
