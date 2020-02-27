#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_reference_arg,
                             assert_inputs_exist)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute bundle volume in mm³',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')

    add_reference_arg(p)
    add_json_args(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundle, optional=args.reference)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

    bundle_name, _ = os.path.splitext(os.path.basename(args.in_bundle))
    stats = {bundle_name: {}}
    if len(sft.streamlines) == 0:
        stats[bundle_name]['volume'] = None
        print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))
        return

    tdi = compute_tract_counts_map(sft.streamlines,
                                   tuple(sft.space_attribute[1]))
    voxel_volume = np.prod(np.prod(sft.space_attribute[2]))
    stats[bundle_name]['volume'] = np.count_nonzero(tdi) * voxel_volume

    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
