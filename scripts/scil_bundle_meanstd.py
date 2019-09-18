#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import os

import nibabel as nib

from scilpy.utils.filenames import split_name_with_nii
from scilpy.io.image import assert_same_resolution
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import assert_inputs_exist, add_reference
from scilpy.utils.metrics_tools import (
    get_metrics_stats_over_streamlines_robust)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute mean and std along the bundle for each metric',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('bundle',
                   help='Fiber bundle file to compute statistics on')

    add_reference(p)

    p.add_argument('metrics', nargs='+',
                   help='Nifti metric(s) to compute statistics on')
    p.add_argument('--density_weighting',
                   action='store_true',
                   help='If set, weight statistics by the number of '
                        'fibers passing through each voxel.')
    p.add_argument('--indent',
                   type=int, default=2,
                   help='Indent for json pretty print.')
    p.add_argument('--sort_keys', action='store_true',
                   help='Sort keys in output json.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.bundle] + args.metrics)

    metrics = [nib.load(metric) for metric in args.metrics]
    assert_same_resolution(*metrics)

    sft = load_tractogram_with_reference(parser, args, args.bundle)
    sft.to_vox()
    sft.to_corner()

    bundle_stats = get_metrics_stats_over_streamlines_robust(
        sft, metrics, args.density_weighting)

    bundle_name, _ = os.path.splitext(os.path.basename(args.bundle))

    stats = {bundle_name: {}}
    for metric, (mean, std) in zip(metrics, bundle_stats):
        metric_name = split_name_with_nii(
            os.path.basename(metric.get_filename()))[0]
        stats[bundle_name][metric_name] = {
            'mean': mean,
            'std': std
        }

    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
