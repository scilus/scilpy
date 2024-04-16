#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracking algorithm conceived to follow and reconstruct fibertubes, with
a given artificial degradation of the resolution during the tracking process.
"""
import os
import time
import argparse
import logging
import numpy as np
import nibabel as nib
import dipy.core.geometry as gm

from scilpy.tracking.seed import FibertubeSeedGenerator
from scilpy.tracking.propagator import FibertubePropagator
from scilpy.image.volume_space_management import FibertubeDataVolume
from scilpy.tracking.fibertube import (add_mandatory_tracking_options,
                                       add_tracking_options,
                                       add_seeding_options,
                                       add_random_options,
                                       add_out_options)
from scilpy.io.utils import save_dictionary
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import save_tractogram
from scilpy.tracking.tracker import Tracker
from scilpy.image.volume_space_management import DataVolume
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_processes_arg,
                             add_verbose_arg,
                             add_reference_arg,
                             add_bbox_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    add_mandatory_tracking_options(p)
    add_tracking_options(p)
    add_seeding_options(p)

    p.add_argument('--single_diameter', action='store_true',
                   help='If set, the first diameter found in \n'
                   '[in_diameters] will be repeated for each fiber.')

    rand_g = add_random_options(p)
    rand_g.add_argument(
        '--skip', type=int, default=0,
        help="Skip the first N seeds. \n"
             "Useful if you want to create new streamlines to "
             "add to \na previously created tractogram with a "
             "fixed --rng_seed.\nEx: If tractogram_1 was created "
             "with -nt 1,000,000, \nyou can create tractogram_2 "
             "with \n--skip 1,000,000.")


    add_processes_arg(p)
    add_out_options(p)
    add_verbose_arg(p)
    add_reference_arg(p)
    add_bbox_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if (args.do_not_compress):
        logging.warning('Streamline compression deactivated. This is not \n'
                        'recommended.')

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('numba').setLevel(logging.WARNING)

    if not nib.streamlines.is_supported(args.in_centroids):
        parser.error('Invalid input streamline file format (must be trk ' +
                     'or tck): {0}'.format(args.in_centroids))

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk ' +
                     'or tck): {0}'.format(args.out_tractogram))

    out_tractogram_no_ext, ext = os.path.splitext(args.out_tractogram)

    outputs = [args.out_tractogram]
    if args.save_seeds:
        outputs.append(out_tractogram_no_ext + '_seeds' + ext)

    assert_inputs_exist(parser, [args.in_centroids, args.in_diameters])
    assert_outputs_exist(parser, args, outputs)

    algo = 'prob'
    theta = gm.math.radians(args.theta)

    max_nbr_pts = int(args.max_length / args.step_size)
    min_nbr_pts = max(int(args.min_length / args.step_size), 1)

    our_space = Space.VOXMM
    our_origin = Origin('center')

    logging.debug('Loading centroid tractogram & diameters')
    in_sft = load_tractogram_with_reference(parser, args, args.in_centroids)
    in_sft.to_voxmm()
    in_sft.to_center()
    # Casting ArraySequence as a list to improve speed
    fibers = list(in_sft.get_streamlines_copy())
    diameters = np.loadtxt(args.in_diameters, dtype=np.float64)
    if args.single_diameter:
        diameters = [diameters[0]]*len(fibers)

    if args.shuffle:
        logging.debug('Shuffling fibers')
        indexes = list(range(len(fibers)))
        gen = np.random.default_rng(args.rng_seed)
        gen.shuffle(indexes)

        new_fibers = []
        new_diameters = []
        for _, index in enumerate(indexes):
            new_fibers.append(fibers[index])
            new_diameters.append(diameters[index])

        fibers = new_fibers
        diameters = new_diameters
        in_sft = StatefulTractogram.from_sft(fibers, in_sft)

    logging.debug("Loading tracking mask.")
    mask_img = nib.load(args.in_mask)
    mask_data = mask_img.get_fdata(caching='unchanged', dtype=float)
    mask_res = mask_img.header.get_zooms()[:3]
    mask = DataVolume(mask_data, mask_res, 'nearest')
    datavolume = FibertubeDataVolume(fibers, diameters, mask_data, mask_res,
                                     args.sampling_radius, our_origin,
                                     np.random.default_rng(args.rng_seed))

    logging.debug("Instantiating seed generator")
    seed_generator = FibertubeSeedGenerator(fibers, diameters,
                                            args.nb_seeds_per_fiber)

    logging.debug("Instantiating propagator")
    propagator = FibertubePropagator(datavolume, args.step_size,
                                     args.rk_order, algo, theta, our_space,
                                     our_origin)

    logging.debug("Instantiating tracker")
    if args.nb_fibers:
        nbr_seeds = args.nb_seeds_per_fiber * args.nb_fibers
    else:
        nbr_seeds = args.nb_seeds_per_fiber * len(fibers)

    compression_th = None if args.do_not_compress else 0

    tracker = Tracker(propagator, mask, seed_generator, nbr_seeds,
                      min_nbr_pts, max_nbr_pts,
                      args.max_invalid_nb_points, compression_th,
                      args.nbr_processes, args.save_seeds, 'r+',
                      rng_seed=args.rng_seed,
                      track_forward_only=args.forward_only,
                      skip=args.skip,
                      verbose=args.verbose,
                      append_last_point=args.keep_last_out_point)

    start_time = time.time()
    logging.debug("Tracking...")
    streamlines, seeds = tracker.track()
    str_time = "%.2f" % (time.time() - start_time)

    logging.debug('Finished tracking in: ' + str_time + ' seconds')

    sft = StatefulTractogram(streamlines, mask_img, our_space,
                             origin=our_origin)
    save_tractogram(sft, args.out_tractogram)

    if args.save_seeds:
        np.savetxt(out_tractogram_no_ext + '_seeds.txt', seeds)

    if args.save_config:
        config = {
            'step_size': args.step_size,
            'sampling_radius': args.sampling_radius,
            'nb_fibers': args.nb_fibers,
            'nb_seeds_per_fiber': args.nb_seeds_per_fiber
        }
        save_dictionary(config, out_tractogram_no_ext + '_config.txt',
                        args.overwrite)


if __name__ == "__main__":
    main()
