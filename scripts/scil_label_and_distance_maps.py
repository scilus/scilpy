#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute assignment map from bundle and centroid streamline.
This script can be very memory hungry on large fiber bundle.
"""

import argparse
import logging

from dipy.io.utils import is_header_compatible
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_reference_arg)
from scilpy.tractanalysis.distance_to_centroid import min_dist_to_centroid


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle', help='Fiber bundle file')
    p.add_argument('in_centroid',
                   help='Centroid streamline associated to input fiber bundle')
    p.add_argument('output_label',
                   help='Output (.npz) file containing the label of the '
                        'nearest point on the centroid streamline for each'
                        ' point of the bundle')
    p.add_argument('output_distance',
                   help='Output (.npz) file containing the distance (in mm) '
                        'to the nearest centroid streamline for each point of '
                        'the bundle')
    add_overwrite_arg(p)
    add_reference_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bundle, args.in_centroid])
    assert_outputs_exist(parser, args, [args.output_label,
                                        args.output_distance])

    is_header_compatible(args.in_bundle, args.in_centroid)

    sft_bundle = load_tractogram_with_reference(parser, args,
                                                args.in_bundle)

    sft_centroid = load_tractogram_with_reference(parser, args,
                                                  args.in_centroid)

    if not len(sft_bundle.streamlines):
        logging.error('Empty bundle file {}. Skipping'
                      .format(args.in_bundle))
        raise ValueError

    if not len(sft_centroid.streamlines):
        logging.error('Empty centroid streamline file {}. Skipping'
                      .format(args.centroid_streamline))
        raise ValueError

    min_dist_label, min_dist = min_dist_to_centroid(sft_bundle.streamlines.data,
                                                    sft_centroid.streamlines.data)
    min_dist_label += 1

    # Save assignment in a compressed numpy file
    # You can load this file and access its data using
    # f = np.load('someFile.npz')
    # assignment = f['arr_0']
    np.savez_compressed(args.output_label, min_dist_label)

    # Save distance in a compressed numpy file
    # You can load this file and access its data using
    # f = np.load('someFile.npz')
    # distance = f['arr_0']
    np.savez_compressed(args.output_distance, min_dist)


if __name__ == '__main__':
    main()
