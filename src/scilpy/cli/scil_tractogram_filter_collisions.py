#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given an input tractogram and a text file containing a diameter for each
streamline, filters all intersecting streamlines and saves the resulting
tractogram and diameters.

IMPORTANT: The input tractogram needs to have been resampled to segments of
at most 0.2mm. Otherwise performance will drop significantly. This is because
this script relies on a KDTree to find all neighboring streamline segments of
any given point. Because the search radius is set at the length of the longest
fibertube segment, the performance drops significantly if they are not
shortened to ~0.2mm.
(see scil_tractogram_resample_nb_points.py)

IMPORTANT: Some tractograms, especially if old, were created with a very high
float precision. scil_tractogram_filter_collisions.py does not save its output
with such precision. This means that after filtering once and saving the
result, new collisions may be created from saving at a lower float precision.
It will require a second filtering to be truly collision-free.

If you are using the --out_metrics parameter on high float precision data, the
script may even throw an error saying that not all collisions were filtered
prior to metrics computation.

Solution: If you encounter such behaviour, we recommend you load and save your
tractogram to be filtered with up-to-date tools such as MI-Brain or the
Nibabel python library. (Which scilpy scripts use)

----------

The filtering is deterministic and follows this approach:
    - Pick next streamline
    - Iterate over its segments
        - If current segment collides with any other streamline segment given
          their diameters
            - Current streamline is deemed invalid and is filtered out
            - Other streamline is left untouched
    - Repeat

This means that the order of the streamlines within the tractogram has a
direct impact on which streamline gets filtered out. To counter the resulting
bias, streamlines are shuffled first unless --disable_shuffling is set.

If the --out_metrics parameter is given, several metrics about the data will
be computed (all expressed in mm):
    - min_external_distance
        Smallest distance separating two streamlines, outside their diameter.
    - max_voxel_anisotropic
        Diagonal vector of the largest possible anisotropic voxel that
        would not intersect two streamlines, given their diameter.
    - max_voxel_isotropic
        Isotropic version of max_voxel_anisotropic made by using the smallest
        component.
        Ex: max_voxel_anisotropic: (3, 5, 5) => max_voxel_isotropic: (3, 3, 3)
    - max_voxel_rotated
        Largest possible isotropic voxel obtainable if the tractogram is
        rotated.
        It is only usable if the entire tractogram is rotated according to
        [rotation_matrix].
        Ex: max_voxel_anisotropic: (1, 0, 0) => max_voxel_isotropic: (0, 0, 0)
            => max_voxel_rotated: (0.5774, 0.5774, 0.5774)

If the --out_rotation_matrix option is provided, the following will be saved:
    - rotation_matrix
        4D transformation matrix representing the rotation to be applied on
        the tractogram to align max_voxel_rotated with the coordinate system
        (see scil_tractogram_apply_transform.py).

See also:
    - docs/source/documentation/fibertube_tracking.rst
"""

import os
import json
import argparse
import logging
import numpy as np

from scilpy.tractograms.intersection_finder import IntersectionFinder
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.tractanalysis.fibertube_scoring import (min_external_distance,
                                                    max_voxels,
                                                    max_voxel_rotated)
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg,
                             add_json_args)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractogram',
                   help='Path to the tractogram file containing the \n'
                   'streamlines (must be .trk).')

    p.add_argument('in_diameters',
                   help='Path to a text file containing a list of \n'
                   'diameters in mm. Each line corresponds \n'
                   'to the identically numbered streamline. \n'
                   'If unsure, refer to the diameters text file of the \n'
                   'DiSCo dataset. If a single diameter is provided, all \n'
                   'streamlines will be given this diameter.')

    p.add_argument('out_tractogram',
                   help='Tractogram output file free of collision (must \n'
                   'be .trk). By default, the diameters will be \n'
                   'saved as data_per_streamline.')

    p.add_argument('--out_colliding_prefix', default=None, type=str,
                   help='Useful for visualization. If set, the script will \n'
                   'produce two other tractograms files containing \n'
                   'colliding streamlines. The first one contains invalid \n'
                   'streamlines that have been filtered out, along with \n'
                   'their collision point as data per streamline. The \n'
                   'second one contains the potentially valid streamlines \n'
                   'that the first tractogram collided with. Note that the \n'
                   'streamlines in the second tractogram may or may not \n'
                   'have been filtered afterwards. \n'
                   'Filenames are derived from [out_colliding_prefix] with \n'
                   '"_invalid" appended for the first tractogram, and \n'
                   '"_obstacle" appended for the second tractogram. You \n'
                   'may include a path in this prefix.')

    p.add_argument('--out_metrics', default=None, type=str,
                   help='If set, metrics about the streamlines and their \n'
                   'diameter will be computed after filtering and saved at \n'
                   'the given location (must be .json).')

    p.add_argument('--out_rotation_matrix', default=None, type=str,
                   help='If set, the transformation required to align the \n'
                   '"max_voxel_rotated" metric with the coordinate system \n'
                   'will be saved at the given location (must be .mat). \n'
                   'This option requires computing all the metrics, even \n'
                   'if --out_metrics is not provided. If it is provided, '
                   'metrics are not computed twice.')

    p.add_argument('--min_distance', default=0, type=float,
                   help='If set, streamlines will be filtered more \n'
                   'aggressively so that even if they don\'t collide, \n'
                   'being below [min_distance] apart (external to their \n'
                   'diameter) will be interpreted as a collision. This \n'
                   'option is the same as filtering with a large diameter \n'
                   'but only saving a small diameter in out_tractogram. \n'
                   '(Value in mm) [%(default)s]')

    p.add_argument('--disable_shuffling', action='store_true',
                   help='If set, no shuffling will be performed before \n'
                   'the filtering process. Streamline segments will be \n'
                   'picked in order from the first segment of the first \n'
                   'streamlines to the last segment of the last streamline.')

    p.add_argument('--rng_seed', type=int, default=0,
                   help='If set, all random values will be generated \n'
                   'using the specified seed. [%(default)s]')

    add_json_args(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    logging.getLogger('numba').setLevel(logging.WARNING)

    in_tractogram_no_ext, in_tractogram_ext = os.path.splitext(
        args.in_tractogram)
    if in_tractogram_ext != '.trk':
        raise ValueError("Invalid output streamline file format " +
                         "(must be trk): {0}".format(args.in_tractogram))

    if os.path.splitext(args.out_tractogram)[1] != '.trk':
        raise ValueError("Invalid output streamline file format " +
                         "(must be trk): {0}".format(args.out_tractogram))

    if args.out_metrics:
        if os.path.splitext(args.out_metrics)[1] != '.json':
            raise ValueError("Invalid metrics output file format " +
                             "(must be json): {0}".format(args.out_metrics))

    if args.out_rotation_matrix:
        if os.path.splitext(args.out_rotation_matrix)[1] != '.mat':
            raise ValueError("Invalid out_rotation_matrix output file" +
                             "format (must be mat): " +
                             "{0}".format(args.out_rotation_matrix))

    outputs = [args.out_tractogram]
    if args.out_colliding_prefix:
        outputs.append(args.out_colliding_prefix + '_invalid.trk')
        outputs.append(args.out_colliding_prefix + '_obstacle.trk')

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, outputs,
                         [args.out_metrics, args.out_rotation_matrix])

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
        raise ValueError('Number of diameters does not match the number ' +
                         'of streamlines.')

    # Casting ArraySequence as a list to improve speed
    streamlines = list(streamlines)

    logging.debug('Building IntersectionFinder')
    inter_finder = IntersectionFinder(
        in_sft, diameters, not args.disable_shuffling, args.rng_seed,
        args.verbose != 'WARNING')

    logging.debug('Finding intersections')
    inter_finder.find_intersections(args.min_distance)

    logging.debug('Building new tractogram(s)')
    out_sft, invalid_sft, obstacle_sft = inter_finder.build_tractograms(
        args.out_colliding_prefix)

    logging.debug('Saving new tractogram(s)')
    save_tractogram(out_sft, args.out_tractogram)

    if args.out_colliding_prefix:
        save_tractogram(
            invalid_sft,
            args.out_colliding_prefix + '_invalid.trk')

        save_tractogram(
            obstacle_sft,
            args.out_colliding_prefix + '_obstacle.trk')

    logging.debug('Input streamline count: ' + str(len(streamlines)) +
                  ' | Output streamline count: ' +
                  str(len(out_sft.streamlines)))

    logging.debug(
        str(len(streamlines) - len(out_sft.streamlines)) +
        ' streamlines have been filtered')

    if args.out_metrics is not None or args.out_rotation_matrix is not None:
        logging.info('Computing metrics')

        min_ext_dist, min_ext_dist_vect = (
            min_external_distance(
                out_sft,
                args.verbose != 'WARNING'))
        max_voxel_ani, max_voxel_iso = max_voxels(min_ext_dist_vect)
        mvr_rot, mvr_edge = max_voxel_rotated(min_ext_dist_vect)

        if args.out_metrics:
            metrics = {
                'min_external_distance': min_ext_dist.tolist(),
                'max_voxel_anisotropic': max_voxel_ani.tolist(),
                'max_voxel_isotropic': max_voxel_iso.tolist(),
                'max_voxel_rotated': [mvr_edge]*3
            }

            with open(args.out_metrics, 'w') as outfile:
                json.dump(metrics, outfile,
                          indent=args.indent, sort_keys=args.sort_keys)

        if args.out_rotation_matrix is not None:
            max_voxel_rotated_transform = np.r_[np.c_[
                mvr_rot, [0, 0, 0]], [[0, 0, 0, 1]]]
            with open(args.out_rotation_matrix, 'w') as outfile:
                np.savetxt(outfile, max_voxel_rotated_transform)


if __name__ == "__main__":
    main()
