#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import os

import nibabel as nib
import numpy as np

from scilpy.io.image import assert_same_resolution
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import assert_inputs_exist
from scilpy.tracking.tools import resample_streamlines
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.metrics_tools import get_metrics_profile_over_streamlines


def norm_l2(x):
    return np.sqrt(np.sum(np.power(x, 2), axis=1, dtype="float"))


def average_euclidean(x, y):
    return np.mean(norm_l2(x - y))


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute tract profiles and their statistics along '
                    'streamlines',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file to compute the tract '
                        'profiles on.')
    p.add_argument('metrics', nargs='+',
                   help='Nifti metric(s) on which to compute '
                        'the tract profiles.')
    p.add_argument('--num_points',
                   type=int, default=20,
                   help='Subsample each streamline to this number of points.')
    p.add_argument('--indent',
                   type=int, default=2,
                   help='Indent for json pretty print.')
    p.add_argument('--sort_keys',
                   action='store_true',
                   help='Sort keys in output json.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bundle, args.metrics])

    if args.num_points <= 1:
        parser.error('--num_points {} needs to be greater than '
                     '1'.format(args.num_points))

    metrics = [nib.load(m) for m in args.metrics]
    assert_same_resolution(*metrics)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)

    bundle_name, _ = os.path.splitext(os.path.basename(args.in_bundle))
    stats = {}
    if len(sft.streamlines) == 0:
        stats[bundle_name] = None
        print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))
        return

    sft.to_vox()
    bundle_subsampled = resample_streamlines(sft.streamline,
                                             num_points=args.num_points,
                                             arc_length=True)

    # Make sure all streamlines go in the same direction. We want to make
    # sure point #1 / 20 of streamline A is matched with point #1 / 20 of
    # streamline B and so on
    num_streamlines = len(bundle_subsampled)
    reference = bundle_subsampled[0]
    for s in np.arange(num_streamlines):
        streamline = bundle_subsampled[s]
        direct = average_euclidean(reference, streamline)
        flipped = average_euclidean(reference, streamline[::-1])

        if flipped < direct:
            bundle_subsampled[s] = streamline[::-1]

    profiles = get_metrics_profile_over_streamlines(bundle_subsampled,
                                                    metrics)
    t_profiles = np.expand_dims(profiles, axis=1)
    t_profiles = np.rollaxis(t_profiles, 3, 2)

    stats[bundle_name] = {}
    for metric, profile, t_profile in zip(metrics, profiles, t_profiles):
        metric_name, _ = split_name_with_nii(
            os.path.basename(metric.get_filename()))
        stats[bundle_name][metric_name] = {
            'mean': np.mean(profile, axis=0).tolist(),
            'std': np.std(profile, axis=0).tolist(),
            'tractprofile': t_profile.tolist()
        }

    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
