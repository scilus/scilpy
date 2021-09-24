#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uniformize streamlines' endpoints according to a defined axis.
Useful for tractometry or models creation.

The --auto option will automatically calculate the main orientation.
If the input bundle is poorly defined, it is possible heuristic will be wrong.

The default is to flip each streamline so their first point's coordinate in the
defined axis is smaller than their last point (--swap does the opposite).
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_outputs_exist,
                             assert_inputs_exist)
from scilpy.utils.streamlines import uniformize_bundle_sft


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Input path of the tractography file.')
    p.add_argument('out_bundle',
                   help='Output path of the uniformized file.')

    method = p.add_mutually_exclusive_group(required=True)
    method.add_argument('--axis', choices=['x', 'y', 'z'],
                        help='Match endpoints of the streamlines along this axis.'
                        '\nSUGGESTION: Commissural = x, Association = y, '
                        'Projection = z')
    method.add_argument('--auto', action='store_true',
                        help='Match endpoints of the streamlines along an '
                             'automatically determined axis.')
    method.add_argument('--centroid', metavar='FILE',
                        help='Match endpoints of the streamlines along an '
                             'automatically determined axis.')
    p.add_argument('--swap', action='store_true',
                   help='Swap head <-> tail convention. '
                        'Can be useful when the reference is not in RAS.')
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, args.in_bundle)
    assert_outputs_exist(parser, args, args.out_bundle)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    if args.auto:
        args.axis = None
    if args.centroid:
        centroid_sft = load_tractogram_with_reference(parser, args,
                                                      args.centroid)
    else:
        centroid_sft = None
    uniformize_bundle_sft(sft, args.axis, ref_bundle=centroid_sft,
                          swap=args.swap)
    save_tractogram(sft, args.out_bundle)


if __name__ == "__main__":
    main()
