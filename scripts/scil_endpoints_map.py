#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
from distutils.version import LooseVersion
import logging
import json
import os

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_trk_in_voxel_space
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)

DESCRIPTION = '''
Computes the endpoint map of a bundle. The endpoint map
is simply a count of the number of streamlines that
start or end in each voxel. The idea is to estimate the
cortical areas affected by the bundle (assuming
streamlines start/end in the cortex)
'''

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('bundle',
                   help='Fiber bundle filename.')
    p.add_argument('reference',
                   help='Reference anatomic filename.')
    p.add_argument('endpoints_map_head',
                   help='Endpoints map head filename.')
    p.add_argument('endpoints_map_tail',
                   help='Endpoints map tail filename.')
    p.add_argument('--indent', type=int, default=2,
                   help='Indent for json pretty print.')
    p.add_argument('--swap', action='store_true',
                   help='Swap head<->tail convention. '
                        'Can be useful when the reference is not in RAS.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    swap = args.swap

    assert_inputs_exist(parser, [args.bundle, args.reference])
    assert_outputs_exist(parser, args, [args.endpoints_map_head,
                                         args.endpoints_map_tail])

    bundle_tractogram_file = nib.streamlines.load(args.bundle)
    if int(bundle_tractogram_file.header['nb_streamlines']) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return

    reference = nib.load(args.reference)
    bundle_streamlines_vox = load_trk_in_voxel_space(
        bundle_tractogram_file, anat=reference)

    endpoints_map_head = np.zeros(reference.shape)
    endpoints_map_tail = np.zeros(reference.shape)

    head_name = args.endpoints_map_head
    tail_name = args.endpoints_map_tail

    if swap:
        head_name = args.endpoints_map_tail
        tail_name = args.endpoints_map_head

    for streamline in bundle_streamlines_vox:
        xyz = streamline[0, :].astype(int)
        endpoints_map_head[xyz[0], xyz[1], xyz[2]] += 1

        xyz = streamline[-1, :].astype(int)
        endpoints_map_tail[xyz[0], xyz[1], xyz[2]] += 1

    nib.save(nib.Nifti1Image(endpoints_map_head, reference.affine,
                             reference.header),
             head_name)
    nib.save(nib.Nifti1Image(endpoints_map_tail, reference.affine,
                             reference.header),
             tail_name)

    bundle_name, _ = os.path.splitext(os.path.basename(args.bundle))
    bundle_name_head = bundle_name + '_head'
    bundle_name_tail = bundle_name + '_tail'

    if swap:
        bundle_name_head = bundle_name + '_tail'
        bundle_name_tail = bundle_name + '_head'

    stats = {
        bundle_name_head: {
            'count': np.count_nonzero(endpoints_map_head)
        },
        bundle_name_tail: {
            'count': np.count_nonzero(endpoints_map_tail)
        }
    }

    print(json.dumps(stats, indent=args.indent))


if __name__ == '__main__':
    main()
