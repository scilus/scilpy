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
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_outputs_exist,
                             assert_inputs_exist)
from scilpy.tractanalysis.features import get_streamlines_centroid


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
                        help='Match endpoints of the streamlines along this axis.\n'
                        'SUGGESTION: Commissural = x, Association = y, '
                        'Projection = z')
    method.add_argument('--auto', action='store_true',
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
    axis = ['x', 'y', 'z']
    if args.auto:
        centroid = get_streamlines_centroid(sft.streamlines, 20)[0]
        main_dir_ends = np.argmax(np.abs(centroid[0] - centroid[-1]))
        main_dir_displacement = np.argmax(np.abs(np.sum(np.gradient(centroid,
                                                                    axis=0),
                                                        axis=0)))
        if main_dir_displacement != main_dir_ends:
            logging.info('Ambiguity in orientation, you should use --axis')
        args.axis = axis[main_dir_displacement]
        logging.info('Orienting endpoints of {} in the {} axis'.format(
            args.in_bundle, args.axis))

    axis_pos = axis.index(args.axis)
    for i in range(len(sft.streamlines)):
        # Bitwise XOR
        if bool(sft.streamlines[i][0][axis_pos] > sft.streamlines[i][-1][axis_pos]) \
                ^ bool(args.swap):
            sft.streamlines[i] = sft.streamlines[i][::-1]
            for key in sft.data_per_point[i]:
                sft.data_per_point[key][i] = sft.data_per_point[key][i][::-1]

    save_tractogram(sft, args.out_bundle)


if __name__ == "__main__":
    main()
