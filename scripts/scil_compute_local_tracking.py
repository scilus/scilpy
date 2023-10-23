#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local streamline HARDI tractography.
The tracking direction is chosen in the aperture cone defined by the
previous tracking direction and the angular constraint.

Algo 'eudx': the peak from the spherical function (SF) most closely aligned
to the previous direction.
Algo 'det': the maxima of the spherical function (SF) the most closely aligned
to the previous direction.
Algo 'prob': a direction drawn from the empirical distribution function defined
from the SF.

The local tracking algorithm can also run on the GPU using the --use_gpu
option (experimental). By default, GPU tracking behaves the same as
DIPY. Below is a list of known divergences between the CPU and GPU
implementations:
    * Backend: The CPU implementation uses DIPY's LocalTracking and the
        GPU implementation uses an in-house OpenCL implementation.
    * Algo: For the GPU implementation, the only available algorithm is
        Algo 'prob'.
    * Tracking sphere: The only sphere available for GPU tracking is
        `repulsion724` and `--sub_sphere` is not available for GPU tracking.
    * SH interpolation: For GPU tracking, SH interpolation can be set to either
        nearest neighbour or trilinear (default). With DIPY, the only available
        method is trilinear.
    * Forward tracking: For GPU tracking, the `--forward_only` flag can be used
        to disable backward tracking. This option isn't available for CPU
        tracking.

NOTE: eudx can be used with pre-computed peaks from fodf as well as
evecs_v1.nii.gz from scil_compute_dti_metrics.py (experimental).

All the input nifti files must be in isotropic resolution.
"""

import argparse
import logging
from time import perf_counter
from typing import Iterable

from dipy.core.sphere import HemiSphere
from dipy.data import get_sphere
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.utils import (get_reference_info,
                           create_tractogram_header)
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamlinespeed import length, compress_streamlines
from dipy.tracking import utils as track_utils
import nibabel as nib
from nibabel.streamlines.tractogram import LazyTractogram, TractogramItem
from nibabel.streamlines import detect_format, TrkFile
import numpy as np
from tqdm import tqdm

from scilpy.reconst.utils import (find_order_from_nb_coeff,
                                  get_b_matrix, get_maximas)
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_sphere_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.tools import get_theta
from scilpy.tracking.utils import (add_mandatory_options_tracking,
                                   add_out_options, add_seeding_options,
                                   add_tracking_options,
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
                         choices=['det', 'prob', 'eudx'],
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


def _get_direction_getter(args):
    odf_data = nib.load(args.in_odf).get_fdata(dtype=np.float32)
    sphere = HemiSphere.from_sphere(get_sphere(args.sphere))\
        .subdivide(args.sub_sphere)
    theta = get_theta(args.theta, args.algo)

    non_zeros_count = np.count_nonzero(np.sum(odf_data, axis=-1))
    non_first_val_count = np.count_nonzero(np.argmax(odf_data, axis=-1))

    if args.algo in ['det', 'prob']:
        if non_first_val_count / non_zeros_count > 0.5:
            logging.warning('Input detected as peaks. Input should be'
                            'fodf for det/prob, verify input just in case.')
        if args.algo == 'det':
            dg_class = DeterministicMaximumDirectionGetter
        else:
            dg_class = ProbabilisticDirectionGetter
        return dg_class.from_shcoeff(
            shcoeff=odf_data, max_angle=theta, sphere=sphere,
            basis_type=args.sh_basis,
            relative_peak_threshold=args.sf_threshold)
    elif args.algo == 'eudx':
        # Code for type EUDX. We don't use peaks_from_model
        # because we want the peaks from the provided sh.
        odf_shape_3d = odf_data.shape[:-1]
        dg = PeaksAndMetrics()
        dg.sphere = sphere
        dg.ang_thr = theta
        dg.qa_thr = args.sf_threshold

        # Heuristic to find out if the input are peaks or fodf
        # fodf are always around 0.15 and peaks around 0.75
        if non_first_val_count / non_zeros_count > 0.5:
            logging.info('Input detected as peaks.')
            nb_peaks = odf_data.shape[-1] // 3
            slices = np.arange(0, 15+1, 3)
            peak_values = np.zeros(odf_shape_3d+(nb_peaks,))
            peak_indices = np.zeros(odf_shape_3d+(nb_peaks,))

            for idx in np.argwhere(np.sum(odf_data, axis=-1)):
                idx = tuple(idx)
                for i in range(nb_peaks):
                    peak_values[idx][i] = np.linalg.norm(
                        odf_data[idx][slices[i]:slices[i+1]], axis=-1)
                    peak_indices[idx][i] = sphere.find_closest(
                        odf_data[idx][slices[i]:slices[i+1]])

            dg.peak_dirs = odf_data
        else:
            logging.info('Input detected as fodf.')
            npeaks = 5
            peak_dirs = np.zeros((odf_shape_3d + (npeaks, 3)))
            peak_values = np.zeros((odf_shape_3d + (npeaks, )))
            peak_indices = np.full((odf_shape_3d + (npeaks, )), -1,
                                   dtype='int')
            b_matrix = get_b_matrix(
                find_order_from_nb_coeff(odf_data), sphere, args.sh_basis)

            for idx in np.argwhere(np.sum(odf_data, axis=-1)):
                idx = tuple(idx)
                directions, values, indices = get_maximas(odf_data[idx],
                                                          sphere, b_matrix,
                                                          args.sf_threshold, 0)
                if values.shape[0] != 0:
                    n = min(npeaks, values.shape[0])
                    peak_dirs[idx][:n] = directions[:n]
                    peak_values[idx][:n] = values[:n]
                    peak_indices[idx][:n] = indices[:n]

            dg.peak_dirs = peak_dirs

        dg.peak_values = peak_values
        dg.peak_indices = peak_indices

        return dg


def tqdm_if_verbose(generator: Iterable, verbose: bool, *args, **kwargs):
    if verbose:
        return tqdm(generator, *args, **kwargs)
    return generator


def _save_tractogram(streamlines_generator, tracts_format, odf_sh_img,
                     total_nb_seeds, args):
    voxel_size = odf_sh_img.header.get_zooms()[0]

    scaled_min_length = args.min_length / voxel_size
    scaled_max_length = args.max_length / voxel_size

    # Tracking is expected to be returned in voxel space, origin `center`.
    def tracks_generator_wrapper():
        for strl, seed in tqdm_if_verbose(streamlines_generator,
                                          verbose=args.verbose,
                                          total=total_nb_seeds,
                                          miniters=int(total_nb_seeds / 100),
                                          leave=False):
            if (scaled_min_length <= length(strl) <= scaled_max_length):
                # Seeds are saved with origin `center` by our own convention.
                # Other scripts (e.g. scil_compute_seed_density_map) expect so.
                dps = {}
                if args.save_seeds:
                    dps['seeds'] = seed

                if args.compress:
                    # compression threshold is given in mm, but we
                    # are in voxel space
                    strl = compress_streamlines(
                        strl, args.compress / voxel_size)

                # TODO: Use nibabel utilities for dealing with spaces
                if tracts_format is TrkFile:
                    # Streamlines are dumped in mm space with
                    # origin `corner`. This is what is expected by
                    # LazyTractogram for .trk files (although this is not
                    # specified anywhere in the doc)
                    strl += 0.5
                    strl *= voxel_size  # in mm.
                else:
                    # Streamlines are dumped in true world space with
                    # origin center as expected by .tck files.
                    strl = np.dot(strl, odf_sh_img.affine[:3, :3]) +\
                        odf_sh_img.affine[:3, 3]

                yield TractogramItem(strl, dps, {})

    tractogram = LazyTractogram.from_data_func(tracks_generator_wrapper)
    tractogram.affine_to_rasmm = odf_sh_img.affine

    filetype = nib.streamlines.detect_format(args.out_tractogram)
    reference = get_reference_info(odf_sh_img)
    header = create_tractogram_header(filetype, *reference)

    # Use generator to save the streamlines on-the-fly
    nib.streamlines.save(tractogram, args.out_tractogram, header=header)


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
            _get_direction_getter(args),
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

        #GPU tracking needs the full sphere
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

    # dump streamlines on-the-fly to file
    _save_tractogram(streamlines_generator, tracts_format,
                     odf_sh_img, total_nb_seeds, args)

    # Final logging
    logging.info('Saved tractogram to {0}.'.format(args.out_tractogram))

    # Total runtime
    logging.info('Total runtime of {0:.2f}s.'.format(perf_counter() - t_init))


if __name__ == '__main__':
    main()
