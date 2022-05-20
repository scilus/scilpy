#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local streamline HARDI tractography using scilpy-only methods -- no dipy (i.e
no cython). The goal of this is to have a python-only version that can be
modified more easily by our team when testing new algorithms and parameters,
and that can be used as parent classes in sub-projects of our lab such as in
dwi_ml.

As in scil_compute_local_tracking:

    The tracking direction is chosen in the aperture cone defined by the
    previous tracking direction and the angular constraint.
    - Algo 'det': the maxima of the spherical function (SF) the most closely
    aligned to the previous direction.
    - Algo 'prob': a direction drawn from the empirical distribution function
    defined from the SF.
    - Algo 'eudx' is not yet available!

Contrary to scil_compute_local_tracking:
    - Input nifti files do not necessarily need to be in isotropic resolution.
    - The script works with asymmetric input ODF.
    - The interpolation for the tracking mask and spherical function can be
      one of 'nearest' or 'trilinear'.
    - Runge-Kutta integration is supported for the step function.

A few notes on Runge-Kutta integration.
    1. Runge-Kutta integration is used to approximate the next tracking
       direction by estimating directions from future tracking steps. This
       works well for deterministic tracking. However, in the context of
       probabilistic tracking, the next tracking directions cannot be estimated
       in advance, because they are picked randomly from a distribution. It is
       therefore recommanded to keep the rk_order to 1 for probabilistic
       tracking.
    2. As a rule of thumb, doubling the rk_order will double the computation
       time in the worst case.

References: [1] Girard, G., Whittingstall K., Deriche, R., and
            Descoteaux, M. (2014). Towards quantitative connectivity analysis:
            reducing tractography biases. Neuroimage, 98, 266-278.
"""
import argparse
import logging
import math
import time

import dipy.core.geometry as gm
import nibabel as nib

from dipy.io.stateful_tractogram import StatefulTractogram, Space, \
                                        set_sft_logger_level
from dipy.io.stateful_tractogram import Origin
from dipy.io.streamline import save_tractogram

from scilpy.io.utils import (add_processes_arg, add_sphere_arg,
                             add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.image.datasets import DataVolume
from scilpy.tracking.propagator import ODFPropagator
from scilpy.tracking.seed import SeedGenerator
from scilpy.tracking.tools import get_theta
from scilpy.tracking.tracker import Tracker
from scilpy.tracking.utils import (add_mandatory_options_tracking,
                                   add_out_options, add_seeding_options,
                                   add_tracking_options,
                                   verify_streamline_length_options,
                                   verify_seed_options)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    add_mandatory_options_tracking(p)

    track_g = add_tracking_options(p)
    track_g.add_argument('--algo', default='prob',
                         choices=['det', 'prob'],
                         help='Algorithm to use [%(default)s]')
    add_sphere_arg(track_g, symmetric_only=False)
    track_g.add_argument('--sfthres_init', metavar='sf_th', type=float,
                         default=0.5, dest='sf_threshold_init',
                         help="Spherical function relative threshold value "
                              "for the \ninitial direction. [%(default)s]")
    track_g.add_argument('--rk_order', metavar="K", type=int, default=1,
                         choices=[1, 2, 4],
                         help="The order of the Runge-Kutta integration used "
                              "for the step function.\n"
                              "For more information, refer to the note in the"
                              " script description. [%(default)s]")
    track_g.add_argument('--max_invalid_length', metavar='MAX', type=float,
                         default=1,
                         help="Maximum length without valid direction, in mm. "
                              "[%(default)s]")
    track_g.add_argument('--forward_only', action='store_true',
                         help="If set, tracks in one direction only (forward) "
                              "given the \ninitial seed. The direction is "
                              "randomly drawn from the ODF.")
    track_g.add_argument('--sh_interp', default='trilinear',
                         choices=['nearest', 'trilinear'],
                         help="Spherical harmonic interpolation: "
                              "nearest-neighbor \nor trilinear. [%(default)s]")
    track_g.add_argument('--mask_interp', default='trilinear',
                         choices=['nearest', 'trilinear'],
                         help="Mask interpolation: nearest-neighbor or "
                              "trilinear. [%(default)s]")

    add_seeding_options(p)

    r_g = p.add_argument_group('Random seeding options')
    r_g.add_argument('--rng_seed', type=int, default=0,
                     help='Initial value for the random number generator. '
                          '[%(default)s]')
    r_g.add_argument('--skip', type=int, default=0,
                     help="Skip the first N random number. \n"
                          "Useful if you want to create new streamlines to "
                          "add to \na previously created tractogram with a "
                          "fixed --rng_seed.\nEx: If tractogram_1 was created "
                          "with -nt 1,000,000, \nyou can create tractogram_2 "
                          "with \n--skip 1,000,000.")

    m_g = p.add_argument_group('Memory options')
    add_processes_arg(m_g)

    add_out_options(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    inputs = [args.in_odf, args.in_seed, args.in_mask]
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, args.out_tractogram)

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress)
    verify_seed_options(parser, args)

    theta = gm.math.radians(get_theta(args.theta, args.algo))

    max_nbr_pts = int(args.max_length / args.step_size)
    min_nbr_pts = int(args.min_length / args.step_size) + 1
    max_invalid_dirs = int(math.ceil(args.max_invalid_length / args.step_size))

    logging.debug("Loading seeding mask.")
    seed_img = nib.load(args.in_seed)
    seed_data = seed_img.get_fdata(caching='unchanged', dtype=float)
    seed_res = seed_img.header.get_zooms()[:3]
    seed_generator = SeedGenerator(seed_data, seed_res)
    if args.npv:
        # toDo. This will not really produce n seeds per voxel, only true
        #  in average.
        nbr_seeds = len(seed_generator.seeds) * args.npv
    elif args.nt:
        nbr_seeds = args.nt
    else:
        # Setting npv = 1.
        nbr_seeds = len(seed_generator.seeds)
    if len(seed_generator.seeds) == 0:
        parser.error('Seed mask "{}" does not have any voxel with value > 0.'
                     .format(args.in_seed))

    logging.debug("Loading tracking mask.")
    mask_img = nib.load(args.in_mask)
    mask_data = mask_img.get_fdata(caching='unchanged', dtype=float)
    mask_res = mask_img.header.get_zooms()[:3]
    mask = DataVolume(mask_data, mask_res, args.mask_interp)

    logging.debug("Loading ODF SH data.")
    odf_sh_img = nib.load(args.in_odf)
    odf_sh_data = odf_sh_img.get_fdata(caching='unchanged', dtype=float)
    odf_sh_res = odf_sh_img.header.get_zooms()[:3]
    dataset = DataVolume(odf_sh_data, odf_sh_res, args.sh_interp)

    logging.debug("Instantiating propagator.")
    propagator = ODFPropagator(
        dataset, args.step_size, args.rk_order, args.algo, args.sh_basis,
        args.sf_threshold, args.sf_threshold_init, theta, args.sphere)

    logging.debug("Instantiating tracker.")
    tracker = Tracker(propagator, mask, seed_generator, nbr_seeds, min_nbr_pts,
                      max_nbr_pts, max_invalid_dirs,
                      compression_th=args.compress,
                      nbr_processes=args.nbr_processes,
                      save_seeds=args.save_seeds,
                      mmap_mode='r+', rng_seed=args.rng_seed,
                      track_forward_only=args.forward_only,
                      skip=args.skip)

    start = time.time()
    logging.debug("Tracking...")
    streamlines, seeds = tracker.track()

    str_time = "%.2f" % (time.time() - start)
    logging.debug("Tracked {} streamlines (out of {} seeds), in {} seconds.\n"
                  "Now saving..."
                  .format(len(streamlines), nbr_seeds, str_time))

    # save seeds if args.save_seeds is given
    data_per_streamline = {'seeds': seeds} if args.save_seeds else {}

    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')

    # Compared with scil_compute_local_tracking, using sft rather than
    # LazyTractogram to deal with space.
    # Contrary to scilpy or dipy, where space after tracking is vox, here
    # space after tracking is voxmm.
    # Smallest possible streamline coordinate is (0,0,0), equivalent of
    # corner origin (TrackVis)
    sft = StatefulTractogram(streamlines, mask_img, Space.VOXMM,
                             Origin.TRACKVIS,
                             data_per_streamline=data_per_streamline)
    save_tractogram(sft, args.out_tractogram)


if __name__ == "__main__":
    main()
