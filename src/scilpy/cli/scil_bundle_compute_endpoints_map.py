#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Computes the endpoints maps of a bundle (head and tail). The endpoints maps are
a count of the number of streamlines that start or end in each voxel.

Note: If the streamlines are not ordered, the head/tail are random and not
really two coherent groups.
    - To get a single endpoint map of all endpoints, without grouping the head
    and tail, use
        >> scil_tractogram_compute_density_map --endpoints_only
    - To order streamlines so that they start and end in the same regions, use
        >> scil_bundle_uniformize_endpoints

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
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_bundle',
                   help='Fiber bundle filename.')
    p.add_argument('endpoints_map_head',
                   help='Output endpoints map head filename.')
    p.add_argument('endpoints_map_tail',
                   help='Output endpoints map tail filename.')
    p.add_argument('--out_json',
                   help='Output JSON file with the number of streamlines '
                        'in each endpoint map. [%(default)s]',
                   default='endpoints_map.json')
    p.add_argument('--swap', action='store_true',
                   help='Swap head<->tail convention. '
                        'Can be useful when the reference is not in RAS.')
    p.add_argument('--binary', action='store_true',
                   help="Save outputs as a binary mask instead of a heat map.")

    distance_g = p.add_argument_group(title='Distance options')
    distance_g.add_argument(
        '--distance', type=int, default=1,
        help="Distance to consider at the extremities of the streamlines. \n"
             "Ex: if --unit is points, a value of 1 means that the first and "
             "last \npoints of the streamlines only are considered."
             "[%(default)s]")
    distance_g.add_argument('--unit', type=str, choices=['points', 'mm'],
                            default='points',
                            help='Unit of the distance. [%(default)s]')

    add_json_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verificatons
    assert_inputs_exist(parser, args.in_bundle, args.reference)
    assert_outputs_exist(parser, args, [args.endpoints_map_head,
                                        args.endpoints_map_tail])

    # Loading
    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    if len(sft.streamlines) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return

    # Processing
    to_mm = args.unit == 'mm'
    endpoints_map_head, endpoints_map_tail = get_head_tail_density_maps(
        sft, point_to_select=args.distance, to_millimeters=to_mm,
        binary=args.binary, swap=args.swap)

    # Saving
    transfo, *_ = sft.space_attributes
    nib.save(nib.Nifti1Image(endpoints_map_head, transfo),
             args.endpoints_map_head)
    nib.save(nib.Nifti1Image(endpoints_map_tail, transfo),
             args.endpoints_map_tail)

    # Printing statistics
    bundle_name, _ = os.path.splitext(os.path.basename(args.in_bundle))
    stats = {
        bundle_name + '_head': {
            'count': np.count_nonzero(endpoints_map_head)
        },
        bundle_name + '_tail': {
            'count': np.count_nonzero(endpoints_map_tail)
        }
    }

    logging.info("{}".format(json.dumps(stats, indent=4)))
    with open(args.out_json, 'w') as outfile:
        json.dump(stats, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == '__main__':
    main()
