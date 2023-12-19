#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local streamline HARDI tractography.
The tracking direction is chosen in the aperture cone defined by the
previous tracking direction and the angular constraint.

Algo 'eudx': select the peak from the spherical function (SF) most closely
aligned to the previous direction, and follow an average of it and the previous
direction [1].

Algo 'det': select the orientation corresponding to the maximum of the
spherical function.

Algo 'prob': select a direction drawn from the empirical distribution function
defined from the SF.

Algo 'ptt': select the propagation direction using Parallel-Transport
Tractography (PTT) framework, see [2] for more details.

NOTE: eudx can be used with pre-computed peaks from fodf as well as
evecs_v1.nii.gz from scil_dti_metrics.py (experimental).

NOTE: If tracking with PTT, the step-size should be smaller than usual,
i.e 0.1-0.2mm or lower. The maximum angle between segments (theta) should
be between 10 and 20 degrees.

The local tracking algorithm can also run on the GPU using the --use_gpu
option (experimental). By default, GPU tracking behaves the same as
DIPY. Below is a list of known divergences between the CPU and GPU
implementations:
    * Backend: The CPU implementation uses DIPY's LocalTracking and the
        GPU implementation uses an in-house OpenCL implementation.
    * Algo: For the GPU implementation, the only available algorithm is
        Algo 'prob'.
    * SH interpolation: For GPU tracking, SH interpolation can be set to either
        nearest neighbour or trilinear (default). With DIPY, the only available
        method is trilinear.
    * Forward tracking: For GPU tracking, the `--forward_only` flag can be used
        to disable backward tracking. This option isn't available for CPU
        tracking.

All the input nifti files must be in isotropic resolution.


References
----------
[1]: Garyfallidis, E. (2012). Towards an accurate brain tractography
[PhD thesis]. University of Cambridge. United Kingdom.

[2]: Aydogan, D. B., & Shi, Y. (2020). Parallel transport tractography.
IEEE transactions on medical imaging, 40(2), 635-647.
"""

import argparse
import logging
from time import perf_counter

from dipy.data import get_sphere
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking import utils as track_utils
import nibabel as nib
from nibabel.streamlines import detect_format, TrkFile
import numpy as np
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_sphere_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.utils import (add_mandatory_options_tracking,
                                   add_out_options, add_seeding_options,
                                   add_tracking_options,
                                   get_theta,
                                   get_direction_getter,
                                   save_tractogram,
                                   verify_streamline_length_options,
                                   verify_seed_options)
from scilpy.tracking.tracker import GPUTacker

# GPU tracking arguments default values
DEFAULT_BATCH_SIZE = 10000
DEFAULT_SH_INTERP = 'trilinear'
DEFAULT_FWD_ONLY = False
DEFAULT_GPU_SPHERE = 'repulsion724'


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    add_mandatory_options_tracking(p)

    track_g = add_tracking_options(p)
    track_g.add_argument('--algo', default='prob',
                         choices=['det', 'prob', 'ptt', 'eudx'],
                         help='Algorithm to use. [%(default)s]')
    add_sphere_arg(track_g, symmetric_only=False)
    track_g.add_argument('--sub_sphere',
                         type=int, default=0,
                         help='Subdivides each face of the sphere into 4^s new'
                              ' faces. [%(default)s]')
    add_seeding_options(p)

    gpu_g = p.add_argument_group('GPU options')
    gpu_g.add_argument('--use_gpu', action='store_true',
                       help='Enable GPU tracking (experimental).')
    gpu_g.add_argument('--sh_interp', default=None,
                       choices=['trilinear', 'nearest'],
                       help='SH image interpolation method. '
                            '[{}]'.format(DEFAULT_SH_INTERP))
    gpu_g.add_argument('--forward_only', action='store_true', default=None,
                       help='Perform forward tracking only.')
    gpu_g.add_argument('--batch_size', default=None, type=int,
                       help='Approximate size of GPU batches (number\n'
                            'of streamlines to track in parallel).'
                            ' [{}]'.format(DEFAULT_BATCH_SIZE))

    out_g = add_out_options(p)

    out_g.add_argument('--seed', type=int,
                       help='Random number generator seed.')

    log_g = p.add_argument_group('Logging options')
    add_verbose_arg(log_g)
    return p


def main():
    # report time for logging/benchmarking
    t_init = perf_counter()
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.use_gpu:
        batch_size = args.batch_size or DEFAULT_BATCH_SIZE
        sh_interp = args.sh_interp or DEFAULT_SH_INTERP
        forward_only = args.forward_only or DEFAULT_FWD_ONLY
        if args.algo != 'prob':
            parser.error('Algo `{}` not supported for GPU tracking. '
                         'Set --algo to `prob` for GPU tracking.'
                         .format(args.algo))
    else:
        if args.batch_size is not None:
            parser.error('Invalid argument --batch_size. '
                         'Set --use_gpu to enable.')
        if args.sh_interp is not None:
            parser.error('Invalid argument --sh_interp. '
                         'Set --use_gpu to enable.')
        if args.forward_only is not None:
            parser.error('Invalid argument --forward_only. '
                         'Set --use_gpu to enable.')

    assert_inputs_exist(parser, [args.in_odf, args.in_seed, args.in_mask])
    assert_outputs_exist(parser, args, args.out_tractogram)

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress)
    verify_seed_options(parser, args)

    tracts_format = detect_format(args.out_tractogram)
    if tracts_format is not TrkFile and args.save_seeds:
        logging.warning("You have selected option --save_seeds but you are "
                        "not saving your tractogram as a .trk file. \n"
                        "   Data_per_point information CANNOT be saved. "
                        "Ignoring.")
        args.save_seeds = False

    logging.debug("Loading masks and finding seeds.")
    mask_img = nib.load(args.in_mask)
    mask_data = get_data_as_mask(mask_img, dtype=bool)

    # Make sure the data is isotropic. Else, the strategy used
    # when providing information to dipy (i.e. working as if in voxel space)
    # will not yield correct results. Tracking is performed in voxel space
    # in both the GPU and CPU cases.
    odf_sh_img = nib.load(args.in_odf)
    if not np.allclose(np.mean(odf_sh_img.header.get_zooms()[:3]),
                       odf_sh_img.header.get_zooms()[0], atol=1e-03):
        parser.error(
            'ODF SH file is not isotropic. Tracking cannot be ran robustly.')

    if args.npv:
        nb_seeds = args.npv
        seed_per_vox = True
    elif args.nt:
        nb_seeds = args.nt
        seed_per_vox = False
    else:
        nb_seeds = 1
        seed_per_vox = True

    voxel_size = odf_sh_img.header.get_zooms()[0]
    vox_step_size = args.step_size / voxel_size
    seed_img = nib.load(args.in_seed)

    if np.count_nonzero(seed_img.get_fdata(dtype=np.float32)) == 0:
        raise IOError('The image {} is empty. '
                      'It can\'t be loaded as '
                      'seeding mask.'.format(args.in_seed))

    # Note. Seeds are in voxel world, center origin.
    # (See the examples in random_seeds_from_mask).
    seeds = track_utils.random_seeds_from_mask(
        seed_img.get_fdata(dtype=np.float32),
        np.eye(4),
        seeds_count=nb_seeds,
        seed_count_per_voxel=seed_per_vox,
        random_seed=args.seed)
    total_nb_seeds = len(seeds)

    if not args.use_gpu:
        # LocalTracking.maxlen is actually the maximum length
        # per direction, we need to filter post-tracking.
        max_steps_per_direction = int(args.max_length / args.step_size)

        streamlines_generator = LocalTracking(
            get_direction_getter(
                args.in_odf, args.algo, args.sphere,
                args.sub_sphere, args.theta, args.sh_basis,
                voxel_size, args.sf_threshold),
            BinaryStoppingCriterion(mask_data),
            seeds, np.eye(4),
            step_size=vox_step_size, max_cross=1,
            maxlen=max_steps_per_direction,
            fixedstep=True, return_all=True,
            random_seed=args.seed,
            save_seeds=True)

    else:  # GPU tracking
        # we'll make our streamlines twice as long,
        # to agree with DIPY's implementation
        max_strl_len = int(2.0 * args.max_length / args.step_size) + 1

        # data volume
        odf_sh = odf_sh_img.get_fdata(dtype=np.float32)

        # GPU tracking needs the full sphere
        sphere = get_sphere(args.sphere).subdivide(args.sub_sphere)

        streamlines_generator = GPUTacker(
            odf_sh, mask_data, seeds,
            vox_step_size, max_strl_len,
            theta=get_theta(args.theta, args.algo),
            sf_threshold=args.sf_threshold,
            sh_interp=sh_interp,
            sh_basis=args.sh_basis,
            batch_size=batch_size,
            forward_only=forward_only,
            rng_seed=args.seed,
            sphere=sphere)

    # save streamlines on-the-fly to file
    save_tractogram(streamlines_generator, tracts_format,
                    odf_sh_img, total_nb_seeds, args.out_tractogram,
                    args.min_length, args.max_length, args.compress,
                    args.save_seeds, args.verbose)

    # Final logging
    logging.info('Saved tractogram to {0}.'.format(args.out_tractogram))

    # Total runtime
    logging.info('Total runtime of {0:.2f}s.'.format(perf_counter() - t_init))


if __name__ == '__main__':
    main()
