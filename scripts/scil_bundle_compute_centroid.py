#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a single bundle centroid, using an 'infinite' QuickBundles threshold.

Formerly: scil_compute_centroid.py
"""

import argparse
import logging

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_verbose_arg,
                             add_reference_arg)
from scilpy.tractanalysis.bundle_operations import get_streamlines_centroid


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')
    p.add_argument('out_centroid',
                   help='Output centroid streamline filename.')
    p.add_argument('--nb_points', type=int, default=20,
                   help='Number of points defining the centroid streamline'
                        '[%(default)s].')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bundle, args.reference)
    assert_outputs_exist(parser, args, args.out_centroid)

    if args.nb_points < 2:
        parser.error('--nb_points {} should be >= 2'.format(args.nb_points))

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)

    centroid_streamlines = get_streamlines_centroid(sft.streamlines,
                                                    args.nb_points)

    sft = StatefulTractogram.from_sft(centroid_streamlines, sft)

    save_tractogram(sft, args.out_centroid)


if __name__ == '__main__':
    main()
