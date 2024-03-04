#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Computes the endpoint map of a bundle. The endpoint map is simply a count of
the number of streamlines that start or end in each voxel.

The idea is to estimate the cortical area affected by the bundle (assuming
streamlines start/end in the cortex).

Note: If the streamlines are not ordered the head/tail are random and not
really two coherent groups. Use the following script to order streamlines:
scil_tractogram_uniformize_endpoints.py

Formerly: scil_compute_endpoints_map.py
"""

import argparse
import logging
import json
import os

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_verbose_arg,
                             add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tractograms.streamline_and_mask_operations import \
    get_head_tail_density_maps


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle filename.')
    p.add_argument('endpoints_map_head',
                   help='Output endpoints map head filename.')
    p.add_argument('endpoints_map_tail',
                   help='Output endpoints map tail filename.')
    p.add_argument('--swap', action='store_true',
                   help='Swap head<->tail convention. '
                        'Can be useful when the reference is not in RAS.')
    p.add_argument('--binary', action='store_true',
                   help="Save outputs as a binary mask instead of a heat map.")
    p.add_argument('--nb_points', type=int, default=1,
                   help="Number of points to consider at the extremities"
                        " of the streamlines. [%(default)s]")

    add_json_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    swap = args.swap
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bundle, args.reference)
    assert_outputs_exist(parser, args, [args.endpoints_map_head,
                                        args.endpoints_map_tail])

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()
    if len(sft.streamlines) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return

    transfo, *_ = sft.space_attributes

    head_name = args.endpoints_map_head
    tail_name = args.endpoints_map_tail

    if swap:
        head_name = args.endpoints_map_tail
        tail_name = args.endpoints_map_head

    endpoints_map_head, endpoints_map_tail = \
        get_head_tail_density_maps(sft, args.nb_points)

    if args.binary:
        endpoints_map_head = (endpoints_map_head > 0).astype(np.int16)
        endpoints_map_tail = (endpoints_map_tail > 0).astype(np.int16)

    nib.save(nib.Nifti1Image(endpoints_map_head, transfo), head_name)
    nib.save(nib.Nifti1Image(endpoints_map_tail, transfo), tail_name)

    bundle_name, _ = os.path.splitext(os.path.basename(args.in_bundle))
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
