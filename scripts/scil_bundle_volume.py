#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import os

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import assert_inputs_exist, add_reference
from scilpy.tractanalysis import compute_tract_counts_map


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute bundle volume in mm³',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')

    add_reference(p)

    p.add_argument('--indent',
                   type=int, default=2,
                   help='Indent for json pretty print. [%(default)s]')
    p.add_argument('--sort_keys',
                   action='store_true',
                   help='Sort keys in output json.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bundle, args.reference])

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

    bundle_name, _ = os.path.splitext(os.path.basename(args.in_bundle))
    stats = {bundle_name: {}}
    if len(sft.streamlines) == 0:
        stats[bundle_name]['volume'] = None
        print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))
        return

    ref_img = nib.load(args.reference)

    tdi = compute_tract_counts_map(sft, ref_img.shape)
    voxel_volume = np.prod(ref_img.header['pixdim'][1:4])
    stats[bundle_name]['volume'] = np.count_nonzero(tdi) * voxel_volume

    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
