#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import logging

import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_reference)
from scilpy.tractometry.distance_to_centroid import min_dist_to_centroid

DESCRIPTION = '''
Compute assignment map from bundle and centroid streamline.
This script can be very memory hungry on large fiber bundle.
'''


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')

    add_reference(p)

    p.add_argument('in_centroid',
                   help='Centroid streamline associated to input '
                        'fiber bundle.')
    p.add_argument('out_label',
                   help='Output (.npz) file containing the label of the '
                   'nearest point on the centroid streamline for each point '
                   'of the bundle.')
    p.add_argument('out_distance',
                   help='Output (.npz) file containing the distance (in mm) to'
                   ' the nearest centroid streamline for each point of '
                   'the bundle.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bundle, args.in_centroid])
    assert_outputs_exist(parser, args,
                         [args.out_label, args.out_distance])

    sft_bundle = load_tractogram_with_reference(parser, args, args.in_bundle)
    bundle_streamlines = sft_bundle.streamlines

    sft_centroid = load_tractogram_with_reference(parser, args,
                                                  args.in_centroid)
    centroid_streamline = sft_centroid.streamlines

    nb_bundle_points = bundle_streamlines.total_nb_rows
    if nb_bundle_points == 0:
        logging.warning('Empty bundle file {}. Skipping'
                        .format(args.in_bundle))
        return

    nb_centroid_points = centroid_streamline.total_nb_rows
    if nb_centroid_points == 0:
        logging.warning('Empty centroid streamline file {}. Skipping'
                        .format(args.in_centroid))
        return

    min_dist_label, min_dist = min_dist_to_centroid(bundle_streamlines.data,
                                                    centroid_streamline.data)
    min_dist_label += 1

    # Save assignment in a compressed numpy file
    # You can load this file and access its data using
    # f = np.load('someFile.npz')
    # assignment = f['arr_0']
    np.savez_compressed(args.out_label, min_dist_label)

    # Save distance in a compressed numpy file
    # You can load this file and access its data using
    # f = np.load('someFile.npz')
    # distance = f['arr_0']
    np.savez_compressed(args.out_distance, min_dist)


if __name__ == '__main__':
    main()
