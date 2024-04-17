#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given an input tractogram and a text file containing a diameter for each
streamline, filters all intersecting streamlines and saves the resulting
tractogram and diameters.

The filtering is deterministic and follows this approach:
    - Pick next streamline
    - Iterate over its segments
        - If current segment collides with any other streamline segment given
          their diameters
            - Deem current streamline a "collider" and filter it out
            - Deem other streamline a "collided" and keep it
    - Repeat

This means that the order of the streamlines within the tractogram will have
an impact on which streamline gets filtered. To counter the resulting bias,
use the "--shuffle" parameter.
"""
import os
import argparse
import logging
import numpy as np

from scilpy.tractograms.intersection_finder import IntersectionFinder
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg,
                             add_reference_arg,
                             add_bbox_arg)
from scilpy.tracking.fibertube import add_random_options


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('in_tractogram',
                   help='Path to the tractogram file containing the \n'
                   'streamlines (must be .trk or .tck).')

    p.add_argument('in_diameters',
                   help='Path to a text file containing a list of \n'
                   'diameters in mm. Each line corresponds \n'
                   'to the identically numbered streamline. \n'
                   'If unsure, refer to the diameters text file of the \n'
                   'DiSCo dataset.')

    p.add_argument('out_tractogram',
                   help='Tractogram output file free of collision (must \n'
                   'be .trk or .tck). Another file (.txt) containing the \n'
                   'diameters will be created using the same filename with \n'
                   '"_diameters" appended.')

    p.add_argument('--single_diameter', action='store_true',
                   help='If set, the first diameter found in \n'
                   '[in_diameters] will be repeated for each fiber.')

    p.add_argument('--save_colliders', action='store_true',
                   help='If set, the script will produce another \n'
                   'tractogram (.trk) containing only streamlines that \n'
                   'have filtered out. Its filename is derived from the \n'
                   'out_tractogram parameter with "_colliders" appended.')

    p.add_argument('--save_collided', action='store_true',
                   help='If set, the script will produce another \n'
                   'tractogram (.trk) containing only valid streamlines \n'
                   'that have been collided by a filtered one. Its file \n'
                   'name is derived from the out_tractogram parameter with \n'
                   '"_collided" appended.')

    metrics_g = p.add_argument_group('Metrics options')
    metrics_g.add_argument('--min_distance', default=None, type=float,
                           help='If set, streamtubes will be filtered more \n'
                           'aggressively so that they are a certain \n'
                           'distance apart. In other words, enforces a \n'
                           'resolution at which the data is void of \n'
                           'partial-volume effect. [%(default)s]')

    add_random_options(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_bbox_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    logging.getLogger('numba').setLevel(logging.WARNING)

    out_tractogram_no_ext, _ = os.path.splitext(args.out_tractogram)

    outputs = [args.out_tractogram]
    if args.save_colliders:
        outputs.append(out_tractogram_no_ext + '_colliders.trk')
    if args.save_collided:
        outputs.append(out_tractogram_no_ext + '_collided.trk')

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, outputs)

    logging.debug('Loading tractogram & diameters')
    in_sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    in_sft.to_voxmm()
    in_sft.to_center()
    # Casting ArraySequence as a list to improve speed
    streamlines = list(in_sft.get_streamlines_copy())
    diameters = np.loadtxt(args.in_diameters, dtype=np.float64)
    if args.single_diameter:
        diameter = diameters if np.ndim(diameters) == 0 else diameters[0]
        diameters = np.full(len(streamlines), diameter)

    if args.shuffle:
        logging.debug('Shuffling streamlines')
        indexes = list(range(len(streamlines)))
        gen = np.random.default_rng(args.rng_seed)
        gen.shuffle(indexes)

        new_streamlines = []
        new_diameters = []
        for _, index in enumerate(indexes):
            new_streamlines.append(streamlines[index])
            new_diameters.append(diameters[index])

        streamlines = new_streamlines
        diameters = np.array(new_diameters)
        in_sft = StatefulTractogram.from_sft(streamlines, in_sft)

    logging.debug('Building IntersectionFinder')
    inter_finder = IntersectionFinder(
        in_sft, diameters, logging.getLevelName(args.verbose) != 'WARNING')

    logging.debug('Finding intersections')
    inter_finder.find_intersections(args.min_distance)

    logging.debug('Building new tractogram(s)')
    out_tractograms, out_diameters = inter_finder.build_tractograms(args)

    logging.debug('Saving new tractogram(s)')
    save_tractogram(out_tractograms[0], args.out_tractogram, args.bbox_check)
    np.savetxt(out_tractogram_no_ext + '_diameters.txt', out_diameters)

    if args.save_colliders:
        save_tractogram(
            out_tractograms[1],
            out_tractogram_no_ext + '_colliders.trk',
            args.bbox_check)

    if args.save_collided:
        save_tractogram(
            out_tractograms[2],
            out_tractogram_no_ext + '_collided.trk',
            args.bbox_check)

    logging.debug('Input streamline count: ' + str(len(streamlines)) +
                  ' | Output streamline count: ' +
                  str(out_tractograms[0]._get_streamline_count()))

    logging.debug(
        str(len(streamlines) - out_tractograms[0]._get_streamline_count()) +
        ' streamlines have been filtered')


if __name__ == "__main__":
    main()
