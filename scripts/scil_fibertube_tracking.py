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

To form fibertubes from a set of streamlines, you can use the scripts:
- scil_tractogram_filter_collisions.py to assign a diameter to each streamline
  and remove all colliding fibertubes.
- scil_tractogram_dps_math.py to assign a diameter without filtering.

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

from scilpy.tracking.seed import FibertubeSeedGenerator, CustomSeedsDispenser
from scilpy.tracking.propagator import FibertubePropagator, ODFPropagator
from scilpy.image.volume_space_management import (FibertubeDataVolume,
                                                  FTODFDataVolume)
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import load_tractogram, save_tractogram
from scilpy.tracking.tracker import Tracker
from scilpy.image.volume_space_management import DataVolume
from scilpy.tractograms.streamline_operations import \
    find_seed_indexes_on_streamlines
from scilpy.io.utils import (parse_sh_basis_arg,
                             load_matrix_in_any_format,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_sh_basis_args,
                             add_sphere_arg,
                             add_processes_arg,
                             add_verbose_arg,
                             add_json_args,
                             add_overwrite_arg)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_fibertubes',
                   help='Path to the tractogram (must be .trk) file \n'
                        'containing fibertubes. They must have their \n'
                        'respective diameter saved as data_per_streamline.')

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
        help='Maximum angle between 2 steps. If the angle is \n'
             'too big, streamline is stopped and the \n'
             'following point is NOT included. [%(default)s]')
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

    ftod_g = p.add_argument_group(
        'ftODF Options',
        'Options required if you want to perform fibertube tracking using\n'
        'fibertube orientation distribution (ftODF).\n'
        'If you\'re not familiar with these options, please refer to the\n'
        'fibertube tracking demo.')
    ftod_g.add_argument('--use_ftODF', action='store_true',
                        help='If set, will build a fibertube orientation\n'
                        'distribution function at each tracking step. It \n'
                        'also allows the use of the scilpy ODFPropagator \n'
                        'instead of FibertubePropagator.')
    ftod_g.add_argument('--algo', default='prob', choices=['det', 'prob'],
                        help='Algorithm to use with ftODF. If ftODF is \n'
                             'NOT used, this argument is not considered. \n'
                             '[%(default)s]')
    add_sphere_arg(ftod_g, symmetric_only=False)
    add_sh_basis_args(ftod_g)
    ftod_g.add_argument('--sh_order',
                        type=int, default=8,
                        help='Spherical harmonics order at which to build'
                             ' ftODF. [%(default)s]')
    ftod_g.add_argument('--sub_sphere',
                        type=int, default=0,
                        help='Subdivides each face of the sphere into 4^s new'
                             ' faces. [%(default)s]')
    ftod_g.add_argument('--sfthres', dest='sf_threshold', metavar='sf_th',
                        type=float, default=0.1,
                        help='Spherical function relative threshold. '
                             '[%(default)s]')
    ftod_g.add_argument('--sfthres_init', metavar='sf_th', type=float,
                        default=0.5, dest='sf_threshold_init',
                        help="Spherical function relative threshold value "
                             "for the \ninitial direction. [%(default)s]")

    seed_group = p.add_argument_group(
        'Seeding options')
    seed_group.add_argument(
        '--nb_seeds_per_fibertube', type=int, default=5,
        help='The number of seeds generated randomly in the first segment \n'
             'of each fibertube. The total amount of streamlines will \n'
             'be [nb_seeds_per_fibertube] * [nb_fibertubes]. [%(default)s]')
    seed_group.add_argument(
        '--nb_fibertubes', type=int,
        help='If set, the script will only track a specified \n'
             'amount of fibers. Otherwise, the entire tractogram \n'
             'will be tracked. The total amount of streamlines \n'
             'will be [nb_seeds_per_fibertube] * [nb_fibertubes].')
    seed_group.add_argument(
        '--local_seeding', default='random', choices=['center', 'random'],
        help='Defines where/how seeds will be placed within a fibertube \n'
             'origin segment. [%(default)s]')
    seed_group.add_argument(
        '--in_custom_seeds', type=str,
        help='Path to a file containing a list of custom seeding \n'
             'coordinates (.txt, .mat or .npy). They should be in \n'
             'VOXMM space. Setting this parameter will ignore all other \n'
             'seeding parameters.')

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
    # Using space and origin in the propagator: vox and center, like
    # in dipy.
    our_space = Space.VOXMM
    our_origin = Origin('center')
    sh_basis, is_legacy = parse_sh_basis_arg(args)

    logging.debug('Loading tractogram & diameters')
    in_sft = load_tractogram(args.in_fibertubes, 'same',
                             to_space=our_space,
                             to_origin=our_origin)
    centerlines = list(in_sft.get_streamlines_copy())
    diameters = np.reshape(in_sft.data_per_streamline['diameters'],
                           len(centerlines))

    # The scilpy Tracker requires a mask for tracking, but fibertube tracking
    # aims to eliminate grids (or masks) in tractography. Instead, the tracking
    # stops when no more fibertubes are detected by the Tracker.

    # Since the scilpy Tracker requires a mask, we provide a fake one that will
    # never interfere.
    fake_mask_data = np.ones(in_sft.dimensions)
    fake_mask = DataVolume(fake_mask_data, in_sft.voxel_sizes, 'nearest')

    if args.use_ftODF:
        logging.debug("Instantiating FTODF datavolume")
        datavolume = FTODFDataVolume(centerlines, diameters, in_sft,
                                     args.blur_radius,
                                     np.random.default_rng(args.rng_seed),
                                     sh_basis, args.sh_order)
    else:
        logging.debug("Instantiating fibertube datavolume")
        datavolume = FibertubeDataVolume(centerlines, diameters, in_sft,
                                         args.blur_radius,
                                         np.random.default_rng(args.rng_seed))

    logging.debug("Instantiating seed generator")
    if args.in_custom_seeds:
        seeds = np.squeeze(load_matrix_in_any_format(args.in_custom_seeds))
        seed_generator = CustomSeedsDispenser(seeds, space=our_space,
                                              origin=our_origin)
        nbr_seeds = len(seeds)
    else:
        seed_generator = FibertubeSeedGenerator(centerlines, diameters,
                                                args.nb_seeds_per_fibertube,
                                                args.local_seeding)

        max_nbr_seeds = args.nb_seeds_per_fibertube * len(centerlines)
        if args.nb_fibertubes:
            if args.nb_fibertubes > len(centerlines):
                raise ValueError("The provided number of seeded fibers" +
                                 "exceeds the number of available fibertubes.")
            else:
                nbr_seeds = args.nb_seeds_per_fibertube * args.nb_fibertubes
        else:
            nbr_seeds = max_nbr_seeds

    if args.use_ftODF:
        logging.debug("Instantiating ODF propagator")
        propagator = ODFPropagator(
            datavolume, args.step_size, args.rk_order, args.algo, sh_basis,
            args.sf_threshold, args.sf_threshold_init, theta, args.sphere,
            sub_sphere=args.sub_sphere,
            space=our_space, origin=our_origin, is_legacy=is_legacy)
    else:
        logging.debug("Instantiating fibertube propagator")
        propagator = FibertubePropagator(datavolume, args.step_size,
                                         args.rk_order, theta, our_space,
                                         our_origin)

    logging.debug("Instantiating tracker")
    if args.skip and nbr_seeds + args.skip > max_nbr_seeds:
        raise ValueError("The number of seeds plus the number of skipped " +
                         "seeds requires more fibertubes than there are " +
                         "available.")
    tracker = Tracker(propagator, fake_mask, seed_generator, nbr_seeds,
                      min_nbr_pts, max_nbr_pts,
                      args.max_invalid_nb_points, 0,
                      args.nbr_processes, True, 'r+',
                      rng_seed=args.rng_seed,
                      track_forward_only=not args.use_ftODF,
                      skip=args.skip,
                      verbose=args.verbose,
                      min_iter=1,
                      append_last_point=args.keep_last_out_point)

    start_time = time.time()
    logging.debug("Tracking...")
    streamlines, seeds = tracker.track()
    str_time = "%.2f" % (time.time() - start_time)
    logging.debug('Finished tracking in: ' + str_time + ' seconds')

    seed_indexes = find_seed_indexes_on_streamlines(seeds, streamlines)

    out_sft = StatefulTractogram.from_sft(streamlines, in_sft)
    out_sft.data_per_streamline['seeds'] = seeds
    out_sft.data_per_streamline['seed_ids'] = seed_indexes
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
