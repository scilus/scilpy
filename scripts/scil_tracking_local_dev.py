#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local streamline HARDI tractography using scilpy-only methods -- no dipy (i.e
no cython). The goal of this is to have a python-only version that can be
modified more easily by our team when testing new algorithms and parameters,
and that can be used as parent classes in sub-projects of our lab such as in
dwi_ml.

WARNING. MUCH SLOWER THAN scil_tracking_local.py. We recommand using multi-
processing with option --nb_processes.

Similar to scil_tracking_local:
    The tracking direction is chosen in the aperture cone defined by the
    previous tracking direction and the angular constraint.
    - Algo 'det': the maxima of the spherical function (SF) the most closely
    aligned to the previous direction.
    - Algo 'prob': a direction drawn from the empirical distribution function
    defined from the SF.

Contrary to scil_tracking_local:
    - Algo 'eudx' is not yet available!
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

Formerly: scil_compute_local_tracking_dev.py
"""
import argparse
import logging
import time

import dipy.core.geometry as gm
import nibabel as nib
import numpy as np

from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.stateful_tractogram import Origin
from dipy.io.streamline import save_tractogram
from nibabel.streamlines import detect_format, TrkFile

from scilpy.io.image import assert_same_resolution
from scilpy.io.utils import (add_processes_arg, add_sphere_arg,
                             add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             parse_sh_basis_arg, verify_compression_th)
from scilpy.image.volume_space_management import DataVolume
from scilpy.tracking.propagator import ODFPropagator
from scilpy.tracking.seed import SeedGenerator
from scilpy.tracking.tracker import Tracker
from scilpy.tracking.utils import (add_mandatory_options_tracking,
                                   add_out_options, add_seeding_options,
                                   add_tracking_options,
                                   get_theta,
                                   verify_streamline_length_options,
                                   verify_seed_options)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    # Options common to both scripts
    add_mandatory_options_tracking(p)
    track_g = add_tracking_options(p)
    add_seeding_options(p)

    # Options only for here.
    track_g.add_argument('--algo', default='prob', choices=['det', 'prob'],
                         help='Algorithm to use. [%(default)s]')
    add_sphere_arg(track_g, symmetric_only=False)
    track_g.add_argument('--sub_sphere',
                         type=int, default=0,
                         help='Subdivides each face of the sphere into 4^s new'
                              ' faces. [%(default)s]')
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
    track_g.add_argument('--max_invalid_nb_points', metavar='MAX', type=float,
                         default=0,
                         help="Maximum number of steps without valid "
                              "direction, \nex: if threshold on ODF or max "
                              "angles are reached.\n"
                              "Default: 0, i.e. do not add points following "
                              "an invalid direction.")
    track_g.add_argument('--forward_only', action='store_true',
                         help="If set, tracks in one direction only (forward) "
                              "given the \ninitial seed. The direction is "
                              "randomly drawn from the ODF.")
    track_g.add_argument('--sh_interp', default='trilinear',
                         choices=['nearest', 'trilinear'],
                         help="Spherical harmonic interpolation: "
                              "nearest-neighbor \nor trilinear. [%(default)s]")
    track_g.add_argument('--mask_interp', default='nearest',
                         choices=['nearest', 'trilinear'],
                         help="Mask interpolation: nearest-neighbor or "
                              "trilinear. [%(default)s]")
    track_g.add_argument(
        '--keep_last_out_point', action='store_true',
        help="If set, keep the last point (once out of the tracking mask) "
             "of \nthe streamline. Default: discard them. This is the default "
             " in \nDipy too. Note that points obtained after an invalid "
             "direction \n(ex when angle is too sharp or sh_threshold not "
             "reached) are \nnever added.")
    track_g.add_argument(
        "--n_repeats_per_seed", type=int, default=1,
        help="By default, each seed position is used only once. This option\n"
             "allows for tracking from the exact same seed n_repeats_per_seed"
             "\ntimes. [%(default)s]")

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
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    inputs = [args.in_odf, args.in_seed, args.in_mask]
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, args.out_tractogram)

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress_th)
    verify_seed_options(parser, args)

    tracts_format = detect_format(args.out_tractogram)
    if tracts_format is not TrkFile:
        logging.warning("You have selected option --save_seeds but you are "
                        "not saving your tractogram as a .trk file. \n"
                        "Data_per_point information CANNOT be saved.\n"
                        "Ignoring.")
        args.save_seeds = False

    theta = gm.math.radians(get_theta(args.theta, args.algo))

    max_nbr_pts = int(args.max_length / args.step_size)
    min_nbr_pts = max(int(args.min_length / args.step_size), 1)

    assert_same_resolution([args.in_mask, args.in_odf, args.in_seed])

    # Choosing our space and origin for this tracking
    # If save_seeds, space and origin must be vox, center. Choosing those
    # values.
    our_space = Space.VOX
    our_origin = Origin('center')

    # Preparing everything
    logging.info("Loading seeding mask.")
    seed_img = nib.load(args.in_seed)
    seed_data = seed_img.get_fdata(caching='unchanged', dtype=float)
    if np.count_nonzero(seed_data) == 0:
        raise IOError('The image {} is empty. '
                      'It can\'t be loaded as '
                      'seeding mask.'.format(args.in_seed))

    seed_res = seed_img.header.get_zooms()[:3]
    seed_generator = SeedGenerator(seed_data, seed_res,
                                   space=our_space, origin=our_origin,
                                   n_repeats=args.n_repeats_per_seed)
    if args.npv:
        # toDo. This will not really produce n seeds per voxel, only true
        #  in average.
        nbr_seeds = len(seed_generator.seeds_vox_corner) * args.npv
    elif args.nt:
        nbr_seeds = args.nt
    else:
        # Setting npv = 1.
        nbr_seeds = len(seed_generator.seeds_vox_corner)
    if len(seed_generator.seeds_vox_corner) == 0:
        parser.error('Seed mask "{}" does not have any voxel with value > 0.'
                     .format(args.in_seed))

    logging.info("Loading tracking mask.")
    mask_img = nib.load(args.in_mask)
    mask_data = mask_img.get_fdata(caching='unchanged', dtype=float)
    mask_res = mask_img.header.get_zooms()[:3]
    mask = DataVolume(mask_data, mask_res, args.mask_interp)

    logging.info("Loading ODF SH data.")
    odf_sh_img = nib.load(args.in_odf)
    odf_sh_data = odf_sh_img.get_fdata(caching='unchanged', dtype=float)
    odf_sh_res = odf_sh_img.header.get_zooms()[:3]
    dataset = DataVolume(odf_sh_data, odf_sh_res, args.sh_interp)

    logging.info("Instantiating propagator.")
    # Converting step size to vox space
    # We only support iso vox for now.
    assert odf_sh_res[0] == odf_sh_res[1] == odf_sh_res[2]
    voxel_size = odf_sh_img.header.get_zooms()[0]
    vox_step_size = args.step_size / voxel_size

    # Using space and origin in the propagator: vox and center, like
    # in dipy.
    sh_basis, is_legacy = parse_sh_basis_arg(args)

    propagator = ODFPropagator(
        dataset, vox_step_size, args.rk_order, args.algo, sh_basis,
        args.sf_threshold, args.sf_threshold_init, theta, args.sphere,
        sub_sphere=args.sub_sphere,
        space=our_space, origin=our_origin, is_legacy=is_legacy)

    logging.info("Instantiating tracker.")
    tracker = Tracker(propagator, mask, seed_generator, nbr_seeds, min_nbr_pts,
                      max_nbr_pts, args.max_invalid_nb_points,
                      compression_th=args.compress_th,
                      nbr_processes=args.nbr_processes,
                      save_seeds=args.save_seeds,
                      mmap_mode='r+', rng_seed=args.rng_seed,
                      track_forward_only=args.forward_only,
                      skip=args.skip,
                      append_last_point=args.keep_last_out_point,
                      verbose=args.verbose)

    start = time.time()
    logging.info("Tracking...")
    streamlines, seeds = tracker.track()

    str_time = "%.2f" % (time.time() - start)
    logging.info("Tracked {} streamlines (out of {} seeds), in {} seconds.\n"
                 "Now saving..."
                 .format(len(streamlines), nbr_seeds, str_time))

    # save seeds if args.save_seeds is given
    # We seeded (and tracked) in vox, center, which is what is expected for
    # seeds.
    if args.save_seeds:
        data_per_streamline = {'seeds': seeds}
    else:
        data_per_streamline = {}

    # Compared with scil_tracking_local, using sft rather than
    # LazyTractogram to deal with space.
    # Contrary to scilpy or dipy, where space after tracking is vox, here
    # space after tracking is voxmm.
    # Smallest possible streamline coordinate is (0,0,0), equivalent of
    # corner origin (TrackVis)
    sft = StatefulTractogram(streamlines, mask_img,
                             space=our_space, origin=our_origin,
                             data_per_streamline=data_per_streamline)
    save_tractogram(sft, args.out_tractogram)


if __name__ == "__main__":
    main()
