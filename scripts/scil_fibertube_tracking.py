#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of the fibertube tracking environment using the
architecture of scil_local_tracking_dev.py.

Contrary to traditional white matter fiber tractography, fibertube
tractography does not rely on a discretized grid of fODFs or peaks. It
directly tracks and reconstructs fibertubes, i.e. streamlines that have an
associated diameter.

When the tracking algorithm is about to select a new direction to propagate
the current streamline, it will build a sphere of radius blur_radius and pick
randomly from all the fibertube segments intersecting with it. The larger the
intersection volume, the more likely a fibertube segment is to be picked and
used as a tracking direction. This makes fibertube tracking inherently
probabilistic.

Possible tracking directions are filtered to respect the aperture cone defined
by the previous tracking direction and the angular constraint.

Seeding is done within the first segment of each fibertube.

For a better understanding of Fibertube Tracking please see:
    - docs/source/documentation/fibertube_tracking.rst
"""

import os
import json
import time
import argparse
import logging
import numpy as np
import nibabel as nib
import dipy.core.geometry as gm

from scilpy.tracking.seed import FibertubeSeedGenerator
from scilpy.tracking.propagator import FibertubePropagator
from scilpy.image.volume_space_management import FibertubeDataVolume
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import load_tractogram, save_tractogram
from scilpy.tracking.tracker import Tracker
from scilpy.image.volume_space_management import DataVolume
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_processes_arg,
                             add_verbose_arg,
                             add_json_args,
                             add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('in_fibertubes',
                   help='Path to the tractogram (must be .trk) file \n'
                        'containing fibertubes. They must be: \n'
                        '1- Void of any collision. \n'
                        '2- With their respective diameter saved \n'
                        'as data_per_streamline. \n'
                        'For both of these requirements, see \n'
                        'scil_tractogram_filter_collisions.py.')

    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')

    track_g = p.add_argument_group('Tracking options')
    track_g.add_argument(
        '--blur_radius', type=float, default=0.1,
        help='Radius of the spherical region from which the \n'
        'algorithm will determine the next direction. \n'
        'A blur_radius within [0.001, 0.5] is recommended. \n'
        '[%(default)s]')
    track_g.add_argument(
        '--step_size', type=float, default=0.1,
        help='Step size of the tracking algorithm, in mm. \n'
        'It is recommended to use the same value as the \n'
        'blur_radius, in the interval [0.001, 0.5] \n'
        'The step_size should never exceed twice the \n'
        'blur_radius. [%(default)s]')
    track_g.add_argument(
        '--min_length', type=float, default=10.,
        metavar='m',
        help='Minimum length of a streamline in mm. '
        '[%(default)s]')
    track_g.add_argument(
        '--max_length', type=float, default=300.,
        metavar='M',
        help='Maximum length of a streamline in mm. '
        '[%(default)s]')
    track_g.add_argument(
        '--theta', type=float, default=60.,
        help='Maximum angle between 2 steps. If the angle is '
             'too big, streamline is \nstopped and the '
             'following point is NOT included.\n'
             '[%(default)s]')
    track_g.add_argument(
        '--rk_order', metavar="K", type=int, default=1,
        choices=[1, 2, 4],
        help="The order of the Runge-Kutta integration used \n"
             'for the step function. \n'
             'For more information, refer to the note in the \n'
             'script description. [%(default)s]')
    track_g.add_argument(
        '--max_invalid_nb_points', metavar='MAX', type=int,
        default=0,
        help='Maximum number of steps without valid \n'
             'direction, \nex: No fibertube intersecting the \n'
             'tracking sphere or max angle is reached.\n'
             'Default: 0, i.e. do not add points following '
             'an invalid direction.')
    track_g.add_argument(
        '--keep_last_out_point', action='store_true',
        help='If set, keep the last point (once out of the \n'
             'tracking mask) of the streamline. Default: discard \n'
             'them. This is the default in Dipy too. \n'
             'Note that points obtained after an invalid direction \n'
             '(based on the propagator\'s definition of invalid) \n'
             'are never added.')

    seed_group = p.add_argument_group(
        'Seeding options',
        'When no option is provided, uses --nb_seeds_per_fibertube 5.')
    seed_group.add_argument(
        '--nb_seeds_per_fibertube', type=int, default=5,
        help='The number of seeds planted in the first segment \n'
             'of each fibertube. The total amount of streamlines will \n'
             'be [nb_seeds_per_fibertube] * [nb_fibertubes]. [%(default)s]')
    seed_group.add_argument(
        '--nb_fibertubes', type=int,
        help='If set, the script will only track a specified \n'
             'amount of fibers. Otherwise, the entire tractogram \n'
             'will be tracked. The total amount of streamlines \n'
             'will be [nb_seeds_per_fibertube] * [nb_fibertubes].')

    rand_g = p.add_argument_group('Random options')
    rand_g.add_argument(
        '--rng_seed', type=int, default=0,
        help='If set, all random values will be generated \n'
        'using the specified seed. [%(default)s]')
    rand_g.add_argument(
        '--skip', type=int, default=0,
        help="Skip the first N seeds. \n"
             "Useful if you want to create new streamlines to "
             "add to \na previously created tractogram with a "
             "fixed --rng_seed.\nEx: If tractogram_1 was created "
             "with -nt 1,000,000, \nyou can create tractogram_2 "
             "with \n--skip 1,000,000.")

    out_g = p.add_argument_group('Output options')
    out_g.add_argument(
        '--out_config', default=None, type=str,
        help='If set, the parameter configuration used for tracking will \n'
        'be saved at the specified location (must be .json). If not given, \n'
        'the config will be printed in the console.')

    add_json_args(out_g)
    add_overwrite_arg(out_g)
    add_processes_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    logging.getLogger('numba').setLevel(logging.WARNING)

    if os.path.splitext(args.in_fibertubes)[1] != '.trk':
        parser.error('Invalid input streamline file format (must be trk):' +
                     '{0}'.format(args.in_fibertubes))

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk ' +
                     'or tck): {0}'.format(args.out_tractogram))

    if args.out_config:
        if os.path.splitext(args.out_config)[1] != '.json':
            parser.error('Invalid output file format (must be json): {0}'
                         .format(args.out_config))

    assert_inputs_exist(parser, [args.in_fibertubes])
    assert_outputs_exist(parser, args, [args.out_tractogram],
                         [args.out_config])

    theta = gm.math.radians(args.theta)

    max_nbr_pts = int(args.max_length / args.step_size)
    min_nbr_pts = max(int(args.min_length / args.step_size), 1)

    our_space = Space.VOXMM
    our_origin = Origin('center')

    logging.debug('Loading tractogram & diameters')
    in_sft = load_tractogram(args.in_fibertubes, 'same', our_space, our_origin)
    centerlines = list(in_sft.get_streamlines_copy())
    diameters = np.reshape(in_sft.data_per_streamline['diameters'],
                           len(centerlines))

    logging.debug("Instantiating datavolumes")
    # The scilpy Tracker requires a mask for tracking, but fibertube tracking
    # aims to eliminate grids (or masks) in tractography. Instead, the tracking
    # stops when no more fibertubes are detected by the Tracker.

    # Since the scilpy Tracker requires a mask, we provide a fake one that will
    # never interfere.
    fake_mask_data = np.ones(in_sft.dimensions)
    fake_mask = DataVolume(fake_mask_data, in_sft.voxel_sizes, 'nearest')
    datavolume = FibertubeDataVolume(centerlines, diameters, in_sft,
                                     args.blur_radius,
                                     np.random.default_rng(args.rng_seed))

    logging.debug("Instantiating seed generator")
    seed_generator = FibertubeSeedGenerator(centerlines, diameters,
                                            args.nb_seeds_per_fibertube)

    logging.debug("Instantiating propagator")
    propagator = FibertubePropagator(datavolume, args.step_size,
                                     args.rk_order, theta, our_space,
                                     our_origin)

    logging.debug("Instantiating tracker")
    max_nbr_seeds = args.nb_seeds_per_fibertube * len(centerlines)
    if args.nb_fibertubes:
        if args.nb_fibertubes > len(centerlines):
            raise ValueError("The provided number of seeded fibers exceeds" +
                             "the number of available fibertubes.")
        else:
            nbr_seeds = args.nb_seeds_per_fibertube * args.nb_fibertubes
    else:
        nbr_seeds = max_nbr_seeds

    if args.skip and nbr_seeds + args.skip > max_nbr_seeds:
        raise ValueError("The number of seeds plus the number of skipped " +
                         "seeds requires more fibertubes than there are " +
                         "available.")
    tracker = Tracker(propagator, fake_mask, seed_generator, nbr_seeds,
                      min_nbr_pts, max_nbr_pts,
                      args.max_invalid_nb_points, 0,
                      args.nbr_processes, True, 'r+',
                      rng_seed=args.rng_seed,
                      track_forward_only=True,
                      skip=args.skip,
                      verbose=args.verbose,
                      append_last_point=args.keep_last_out_point)

    start_time = time.time()
    logging.debug("Tracking...")
    streamlines, seeds = tracker.track()
    str_time = "%.2f" % (time.time() - start_time)
    logging.debug('Finished tracking in: ' + str_time + ' seconds')

    out_sft = StatefulTractogram.from_sft(streamlines, in_sft)
    out_sft.data_per_streamline['seeds'] = seeds
    save_tractogram(out_sft, args.out_tractogram)

    config = {
        'step_size': args.step_size,
        'blur_radius': args.blur_radius,
        'nb_fibertubes': (args.nb_fibertubes if args.nb_fibertubes
                          else len(centerlines)),
        'nb_seeds_per_fibertube': args.nb_seeds_per_fibertube
    }
    if args.out_config:
        with open(args.out_config, 'w') as outfile:
            json.dump(config, outfile,
                      indent=args.indent, sort_keys=args.sort_keys)
    else:
        print('Config:\n',
              json.dumps(config, indent=args.indent,
                         sort_keys=args.sort_keys))


if __name__ == "__main__":
    main()
