#!/usr/bin/env python3
"""
Perform anatomically-constrainedtractography for a tracer experiment (e.g.
from Allen Mouse Brain Connectivity Atlas). The script takes as input a WM mask and
projection density map for a tracer injection coregistered to the same space as a
diffusion MRI SH volume.
"""
import argparse
import time
import logging
import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from skimage.morphology import binary_dilation
from skimage.morphology.gray import dilation as gray_dilation
from scilpy.io.image import assert_same_resolution
from dipy.io.streamline import save_tractogram
from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
from dipy.core.sphere import Sphere
from scilpy.io.utils import (add_sphere_arg, add_verbose_arg, add_sh_basis_args,
                             assert_inputs_exist, parse_sh_basis_arg, verify_compression_th,
                             assert_outputs_exist, assert_headers_compatible)
from scilpy.tracking.utils import (get_theta, verify_streamline_length_options, add_mandatory_options_tracking,
                                   verify_seed_options, add_seeding_options, add_out_options)
from scilpy.image.volume_space_management import DataVolume
from scilpy.tracking.seed import SeedGenerator
from scilpy.tracking.propagator import ODFPropagator
from scilpy.tracking.tracker import MouseTracker
from scilpy.version import version_string


# Default values for tracking parameters
ALGO = 'prob'
RK_ORDER = 1


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)
    add_mandatory_options_tracking(p, fodf_optional=False)
    p.add_argument('--peaks_seeding',
                   help='Optional peaks file to use to initialize seeding.')
    p.add_argument('--min_length', default=2.0, type=float,
                   help='Minimum length of streamlines in mm [%(default)s].')
    p.add_argument('--max_length', default=20.0, type=float,
                   help='Maximum length of streamlines in mm [%(default)s].')

    p.add_argument('--theta', default=20, type=float,
                   help='Theta angle in degrees for probabilistic tracking [%(default)s].')
    p.add_argument('--step', default=0.020, type=float, dest='step_size',
                   help='Step size for tracking in mm [%(default)s].')
    p.add_argument('--sfthres', dest='sf_threshold', metavar='sf_th',
                   type=float, default=0.1,
                   help='Spherical function relative threshold [%(default)s].')
    p.add_argument('--sfthres_init', metavar='sf_th', type=float,
                   default=0.5, dest='sf_threshold_init',
                   help="Spherical function relative threshold value ")
    p.add_argument('--max_invalid_nb_points', metavar='MAX', type=float, default=0,
                   help="Maximum number of steps without valid "
                        "direction, \nex: if threshold on ODF or max "
                        "angles are reached.\n"
                        "Default: 0, i.e. do not add points following "
                        "an invalid direction.")
    p.add_argument('--sigma_backtrack', default=4.0, type=float,
                   help='Sigma for the Gaussian distribution used for backtracking, in mm [%(default)s].')
    p.add_argument('--backtrack_length', default=0.500, type=float,
                   help='Distance for backtracking, in mm [%(default)s].')
    p.add_argument('--forward_only', action='store_true',
                   help='If set, only forward tracking will be performed.\n'
                        'By default, both forward and backward tracking are performed.')

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

    add_seeding_options(p)
    add_out_options(p)
    add_verbose_arg(p)

    add_sh_basis_args(p)
    add_sphere_arg(p, symmetric_only=False)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    inputs = [args.in_odf, args.in_mask, args.in_seed]
    assert_inputs_exist(parser, inputs, args.peaks_seeding)
    assert_outputs_exist(parser, args, args.out_tractogram)

    assert_same_resolution(inputs)
    assert_headers_compatible(parser, inputs, args.peaks_seeding)

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))
    
    verify_seed_options(parser, args)
    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress_th)

    theta = np.deg2rad(get_theta(args.theta, 'prob'))
    max_nbr_pts = int(args.max_length / args.step_size)
    min_nbr_pts = max(int(args.min_length / args.step_size), 1)

    # Choosing our space and origin for this tracking
    # If save_seeds, space and origin must be vox, center. Choosing those
    # values.
    our_space = Space.VOX
    our_origin = Origin('center')

    logging.debug('Instantiating SeedGenerator...')
    seed_img = nib.load(args.in_seed)
    seed_data = seed_img.get_fdata(caching='unchanged', dtype=float)
    if np.count_nonzero(seed_data) == 0:
        raise IOError('The image {} is empty. '
                      'It can\'t be loaded as '
                      'seeding mask.'.format(args.in_seed))

    seed_res = seed_img.header.get_zooms()[:3]
    seed_generator = SeedGenerator(seed_data, seed_res, space=our_space, origin=our_origin)

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
        parser.error('Seed mask "{}" does not have any voxel with'
                    ' value > 0.'.format(args.in_seed))

    logging.debug('Number of seeds to be generated: {}'.format(nbr_seeds))

    logging.debug('Loading WM mask...')
    mask_img = nib.load(args.in_mask)
    mask_data = mask_img.get_fdata(caching='unchanged', dtype=float)
    mask_res = mask_img.header.get_zooms()[:3]
    mask = DataVolume(mask_data, mask_res, 'nearest')

    logging.debug('Loading SH volume...')
    odf_img = nib.load(args.in_odf)
    odf_data = odf_img.get_fdata(caching='unchanged', dtype=float)
    sh_basis, is_legacy = parse_sh_basis_arg(args)
    sh_order = order_from_ncoef(odf_data.shape[3])

    if args.peaks_seeding:
        logging.debug('Loading peaks for seeding...')
        assert_same_resolution([args.in_odf, args.peaks_seeding])
        peaks_img = nib.load(args.peaks_seeding)
        peaks_data = peaks_img.get_fdata(caching='unchanged', dtype=float)
        # convert peaks to SH diracs (only inside seeding mask)
        seed_mask = seed_data > 0
        peak_directions = peaks_data[seed_mask]
        peak_norm = np.linalg.norm(peak_directions, axis=-1)
        peak_directions[peak_norm > 0] /= peak_norm[peak_norm > 0][:, None]
        B, _ = sh_to_sf_matrix(Sphere(xyz=peak_directions), sh_order_max=sh_order,
                               basis_type=sh_basis, legacy=is_legacy)
        sh_to_replace = odf_data[seed_mask]
        sf_max = np.max(sh_to_replace.dot(B))
        odf_data[seed_mask] += B.T * sf_max * 0.5
        # save the new ODF data to a temporary file for debugging purposes
        nib.save(nib.Nifti1Image(odf_data, odf_img.affine, odf_img.header), 'temp_odf_with_peaks.nii.gz')

    # Instantiate the ODF datavolume
    voxel_size = odf_img.header.get_zooms()[0]
    odf_datavolume = DataVolume(odf_data, seed_res, 'trilinear', False)

    vox_step_size = args.step_size / voxel_size
    backtrack_nb_pts = int(args.backtrack_length / args.step_size)
    sigma_backtrack_pts = args.sigma_backtrack / args.step_size

    logging.debug('Instantiating ODFPropagatorWithSETPriors...')
    propagator = ODFPropagator(odf_datavolume, vox_step_size,
                               RK_ORDER, ALGO, sh_basis,
                               args.sf_threshold,
                               args.sf_threshold_init, theta,
                               space=our_space, origin=our_origin,
                               is_legacy=is_legacy)
    
    logging.debug('Instantiating MouseTracker...')
    tracker = MouseTracker(propagator, mask, seed_generator,
                            nbr_seeds, min_nbr_pts, max_nbr_pts,
                            args.max_invalid_nb_points, args.compress_th,
                            save_seeds=args.save_seeds, rng_seed=args.rng_seed,
                            track_forward_only=args.forward_only, skip=args.skip,
                            verbose=args.verbose, backtrack_nb_pts=backtrack_nb_pts,
                            sigma_backtrack_pts=sigma_backtrack_pts)

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
    sft = StatefulTractogram(streamlines, seed_img,
                             space=our_space, origin=our_origin,
                             data_per_streamline=data_per_streamline)
    save_tractogram(sft, args.out_tractogram)


if __name__ == "__main__":
    main()
