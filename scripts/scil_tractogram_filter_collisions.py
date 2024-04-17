#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given an input tractogram and a text file containing a diameter for each
streamline, filters all intersecting ones and saves the resulting
tractogram and diameters.
"""
import os
import argparse
import logging
import nibabel as nib
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
                   'to the identically numbered streamline.')

    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or \n'
                   '.tck). Another file (.txt) containing the diameters \n'
                   'will be created using the same file name with \n'
                   '"_diameters" appended.')

    p.add_argument('--single_diameter', action='store_true',
                   help='If set, the first diameter found in \n'
                   '[in_diameters] will be repeated for each fiber.')

    p.add_argument('-cr', '--save_colliders', action='store_true',
                   help='If set, the script will produce another \n'
                   'tractogram (.trk) containing only streamlines that \n'
                   'have filtered out (with their collision point). \n'
                   'Its file name is derived from the out_tractogram \n'
                   'parameter with "_colliders" appended.')

    p.add_argument('-cd', '--save_collided', action='store_true',
                   help='If set, the script will produce another \n'
                   'tractogram (.trk) containing only valid streamlines \n'
                   'that have been collided by a filtered one. Its file \n'
                   'name is derived from the out_tractogram parameter with \n'
                   '"_collided" appended.')

    metrics_g = p.add_argument_group('Metrics options')
    metrics_g.add_argument('--tmv_size_threshold', default=None, type=float,
                           help='If set, fibers that don\'t collide \n'
                           'will be filtered further, so that the edge \n'
                           'length of true_max_voxel is below the given \n'
                           'threshold. (see ft_fibers_metrics.py)'
                           '[%(default)s]')

    add_random_options(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_bbox_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('numba').setLevel(logging.WARNING)

    if not nib.streamlines.is_supported(args.in_tractogram):
        parser.error('Invalid input streamline file format (must be trk ' +
                     'or tck): {0}'.format(args.in_tractogram))

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk ' +
                     'or tck): {0}'.format(args.out_tractogram))

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
    fibers = list(in_sft.get_streamlines_copy())
    diameters = np.loadtxt(args.in_diameters, dtype=np.float64)
    if args.single_diameter:
        diameter = diameters if np.ndim(diameters) == 0 else diameters[0]
        diameters = np.full(len(fibers), diameter)

    if args.shuffle:
        logging.debug('Shuffling streamlines')
        indexes = list(range(len(fibers)))
        gen = np.random.default_rng(args.rng_seed)
        gen.shuffle(indexes)

        new_fibers = []
        new_diameters = []
        for _, index in enumerate(indexes):
            new_fibers.append(fibers[index])
            new_diameters.append(diameters[index])

        fibers = new_fibers
        diameters = np.array(new_diameters)
        in_sft = StatefulTractogram.from_sft(fibers, in_sft)

    print(diameters[:5])

    logging.debug('Building IntersectionFinder')
    inter_finder = IntersectionFinder(in_sft, diameters, args.verbose)

    logging.debug('Finding intersections')
    inter_finder.find_intersections(args.tmv_size_threshold)

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

    logging.debug('Input streamline count: ' + str(len(fibers)) +
                  ' | Output streamline count: ' +
                  str(out_tractograms[0]._get_streamline_count()))

    logging.debug(
        str(len(fibers) - out_tractograms[0]._get_streamline_count()) +
        ' streamlines have been filtered')


if __name__ == "__main__":
    main()
