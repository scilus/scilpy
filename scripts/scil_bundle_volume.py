#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import os

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_trk_in_voxel_space
from scilpy.io.utils import assert_inputs_exist
from scilpy.tractanalysis import compute_robust_tract_counts_map


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Compute bundle volume in mm³',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('bundle',
                        help='Fiber bundle file.')
    parser.add_argument('reference',
                        help='Nifti reference image.')
    parser.add_argument('--indent',
                        type=int, default=2,
                        help='Indent for json pretty print. [%(default)s]')
    parser.add_argument('--sort_keys',
                        action='store_true',
                        help='Sort keys in output json.')

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.bundle, args.reference])

    bundle_tractogram_file = nib.streamlines.load(args.bundle)

    bundle_name, _ = os.path.splitext(os.path.basename(args.bundle))
    stats = {bundle_name: {}}
    if len(bundle_tractogram_file.streamlines) == 0:
        stats[bundle_name]['volume'] = None
        print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))
        return

    ref_img = nib.load(args.reference)
    bundle_streamlines_vox = load_trk_in_voxel_space(
        bundle_tractogram_file, anat=ref_img)
    tdi = compute_robust_tract_counts_map(
        bundle_streamlines_vox, ref_img.shape)
    voxel_volume = np.prod(ref_img.header['pixdim'][1:4])
    stats[bundle_name]['volume'] = np.count_nonzero(tdi) * voxel_volume

    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
