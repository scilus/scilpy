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
            - Current streamline becomes is filtered out of the streamlines
            - Other streamline is left untouched
    - Repeat

This means that the order of the streamlines within the tractogram has a
direct impact on which streamline gets filtered out. To counter the resulting
bias, streamlines are shuffled first unless --disable_shuffling is set.

If the --out_metrics parameter is given, several metrics about the data will be
computed.

Computed metrics:
    - min_external_distance
        Smallest distance separating two fibers.
    - max_voxel_anisotropic
        Diagonal vector of the largest possible anisotropic voxel that
        would not intersect two fibers.
    - max_voxel_isotropic
        Isotropic version of max_voxel_anisotropic made by using the smallest
        component.
        Ex: max_voxel_anisotropic: (3, 5, 5)
            max_voxel_isotropic: (3, 3, 3)
    - max_voxel_rotated
        Largest possible isotropic voxel if the tractogram is rotated. It is
        obtained by measuring the smallest distance between two fibertubes.
        It is only usable if the entire tractogram is rotated according to
        [rotation_matrix].
        Ex: max_voxel_anisotropic: (1, 0, 0)
            max_voxel_isotropic: (0, 0, 0)
            max_voxel_rotated: (0.5774, 0.5774, 0.5774)
    - rotation_matrix [separate file]
        4D transformation matrix representing the rotation to be applied on
        the tractogram to align max_voxel_rotated with the coordinate system
        (see scil_tractogram_apply_transform.py).
"""

import os
import argparse
import logging
import numpy as np

from scilpy.tractograms.intersection_finder import IntersectionFinder
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.tractanalysis.fibertube_scoring import (
    min_external_distance, max_voxels, max_voxel_rotated)
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg,
                             add_reference_arg,
                             add_bbox_arg)
from scilpy.io.utils import save_dictionary


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
                   'be .trk). By default, the diameters will be \n'
                   'saved as data-per-streamline. To turn it off, use the \n'
                   '--no_diameters option')

    p.add_argument('--save_colliding', action='store_true',
                   help='Useful for visualization. If set, the script will \n'
                   'produce two other tractograms (.trk) containing \n'
                   'colliding streamlines. The first one contains all \n'
                   'streamlines that have been filtered out, and the \n'
                   'second one contains the streamlines that the first \n'
                   'tractogram was colliding with. Note that the \n'
                   'streamlines in the second tractogram may or may not \n'
                   'have been filtered afterwards. \n'
                   'Filenames are derived from [in_tractogram] with \n'
                   '"_invalid" appended for the first tractogram, and \n'
                   '"_obstacle" appended for the second tractogram.')

    p.add_argument('--out_metrics', default=None, type=str,
                   help='If set, metrics about the fibertubes will be \n'
                   'computed after filtering and saved at the given \n'
                   'location (must be .txt). Additionally, the \n'
                   'transformation required to align the \n'
                   '"max_voxel_rotated" measure with the coordinate system \n'
                   'will be saved separately under the same filename with \n'
                   '"_max_voxel_rotation" appended.')

    p.add_argument('--min_distance', default=None, type=float,
                   help='If set, streamtubes will be filtered more \n'
                   'aggressively so that they are a certain \n'
                   'distance apart. In other words, enforces a \n'
                   'resolution at which the data is void of \n'
                   'partial-volume effect. [%(default)s]')

    p.add_argument('--disable_shuffling', action='store_true',
                   help='If set, no shuffling will be performed before \n'
                   'the filtering process. Streamlines will be picked in \n'
                   'order.')

    p.add_argument('--rng_seed', type=int, default=0,
                   help='If set, all random values will be generated \n'
                   'using the specified seed. [%(default)s]')

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

    in_tractogram_no_ext, _ = os.path.splitext(args.in_tractogram)
    _, out_tractogram_ext = os.path.splitext(args.out_tractogram)
    if args.out_metrics is not None:
        out_metrics_no_ext, _ = os.path.splitext(args.out_metrics)

    if out_tractogram_ext.lower() != '.trk':
        raise ValueError("Invalid output streamline file format " +
                         "(must be trk): {0}".format(args.tractogram_filename))

    outputs = [args.out_tractogram]
    if args.save_colliding:
        outputs.append(in_tractogram_no_ext + '_invalid.trk')
        outputs.append(in_tractogram_no_ext + '_obstacle.trk')

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, outputs, [args.out_metrics])

    logging.debug('Loading tractogram & diameters')
    in_sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    in_sft.to_voxmm()
    in_sft.to_center()

    streamlines = in_sft.get_streamlines_copy()
    diameters = np.loadtxt(args.in_diameters, dtype=np.float64)

    # Test single diameter
    if np.ndim(diameters) == 0:
        diameters = np.full(len(streamlines), diameters)
    elif diameters.shape[0] != (len(streamlines)):
        raise ValueError('Number of diameters does not match the number' +
                         'of streamlines.')

    if not args.disable_shuffling:
        logging.debug('Shuffling streamlines')
        indexes = list(range(len(streamlines)))
        gen = np.random.default_rng(args.rng_seed)
        gen.shuffle(indexes)

        streamlines = streamlines[indexes]
        diameters = diameters[indexes]
        in_sft = StatefulTractogram.from_sft(streamlines, in_sft)

    # Casting ArraySequence as a list to improve speed
    streamlines = list(streamlines)

    logging.debug('Building IntersectionFinder')
    inter_finder = IntersectionFinder(
        in_sft, diameters, logging.getLevelName(args.verbose) != 'WARNING')

    logging.debug('Finding intersections')
    inter_finder.find_intersections(args.min_distance)

    logging.debug('Building new tractogram(s)')
    out_sft, invalid_sft, obstacle_sft = inter_finder.build_tractograms(
        args.save_colliding)

    logging.debug('Saving new tractogram(s)')
    save_tractogram(out_sft, args.out_tractogram, args.bbox_check)

    if args.save_colliding:
        save_tractogram(
            invalid_sft,
            in_tractogram_no_ext + '_invalid.trk',
            args.bbox_check)

        save_tractogram(
            obstacle_sft,
            in_tractogram_no_ext + '_obstacle.trk',
            args.bbox_check)

    logging.debug('Input streamline count: ' + str(len(streamlines)) +
                  ' | Output streamline count: ' +
                  str(len(out_sft.streamlines)))

    logging.debug(
        str(len(streamlines) - len(out_sft.streamlines)) +
        ' streamlines have been filtered')

    if args.out_metrics is not None:
        logging.info('Computing metrics')
        min_ext_dist, min_ext_dist_vect = (
            min_external_distance(out_sft.streamlines,
                                  out_sft.data_per_streamline['diameters'],
                                  logging.getLevelName(args.verbose) != 'WARNING'))
        max_voxel_ani, max_voxel_iso = max_voxels(min_ext_dist_vect)
        mvr_rot, mvr_edge = max_voxel_rotated(min_ext_dist_vect)

        metrics = {
            'min_external_distance': min_ext_dist,
            'max_voxel_anisotropic': max_voxel_ani,
            'max_voxel_isotropic': max_voxel_iso,
            'max_voxel_rotated': [mvr_edge]*3
        }
        save_dictionary(metrics, args.out_metrics, args.overwrite)

        max_voxel_rotated_transform = np.r_[np.c_[
            mvr_rot, [0, 0, 0]], [[0, 0, 0, 1]]]
        np.savetxt(out_metrics_no_ext + '_max_voxel_rotation.txt',
                    max_voxel_rotated_transform)



if __name__ == "__main__":
    main()
