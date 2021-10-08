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

For streamline compression, a rule of thumb is to set it to 0.1mm for the
deterministic algorithm and 0.2mm for probabilitic algorithm.

NOTE: eudx can be used with pre-computed peaks from fodf as well as
evecs_v1.nii.gz from scil_compute_dti_metrics.py (experimental).

All the input nifti files must be in isotropic resolution.
"""

import argparse
import logging

from dipy.core.sphere import HemiSphere
from dipy.data import SPHERE_FILES, get_sphere
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
from dipy.direction.peaks import PeaksAndMetrics
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
import nibabel as nib
import numpy as np

from scilpy.reconst.utils import (find_order_from_nb_coeff,
                                  get_b_matrix, get_maximas)
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             add_verbose_arg)
from scilpy.tracking.tools import get_theta
from scilpy.tracking.utils import (load_mask_and_verify_anisotropy,
                                   prepare_seeds, verify_tracking_args,
                                   save_results)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p._optionals.title = 'Generic options'
    p.add_argument('in_sh',
                   help='Spherical harmonic file (.nii.gz) OR \n'
                        'peaks/evecs (.nii.gz) for EUDX tracking.')
    p.add_argument('in_seed',
                   help='Seeding mask  (.nii.gz).')
    p.add_argument('in_mask',
                   help='Seeding mask (.nii.gz).\n'
                        'Tracking will stop outside this mask.')
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')

    track_g = p.add_argument_group('Tracking options')
    track_g.add_argument('--algo', default='prob',
                         choices=['det', 'prob', 'eudx'],
                         help='Algorithm to use [%(default)s]')
    track_g.add_argument('--step', dest='step_size', type=float, default=0.5,
                         help='Step size in mm. [%(default)s]')
    track_g.add_argument('--min_length', type=float, default=10.,
                         help='Minimum length of a streamline in mm. '
                              '[%(default)s]')
    track_g.add_argument('--max_length', type=float, default=300.,
                         help='Maximum length of a streamline in mm. '
                              '[%(default)s]')
    track_g.add_argument('--theta', type=float,
                         help='Maximum angle between 2 steps.\n'
                              '["eudx"=60, "det"=45, "prob"=20]')
    track_g.add_argument('--sfthres', dest='sf_threshold',
                         type=float, default=0.1,
                         help='Spherical function relative threshold. '
                              '[%(default)s]')
    add_sh_basis_args(track_g)

    seed_group = p.add_argument_group(
        'Seeding options',
        'When no option is provided, uses --npv 1.')
    seed_sub_exclusive = seed_group.add_mutually_exclusive_group()
    seed_sub_exclusive.add_argument('--npv', type=int,
                                    help='Number of seeds per voxel.')
    seed_sub_exclusive.add_argument('--nt', type=int,
                                    help='Total number of seeds to use.')

    p.add_argument('--sphere', choices=sorted(SPHERE_FILES.keys()),
                   default='symmetric724',
                   help='Set of directions to be used for tracking.')

    out_g = p.add_argument_group('Output options')
    out_g.add_argument('--compress', type=float,
                       help='If set, will compress streamlines.\n'
                            'The parameter value is the distance threshold.')
    out_g.add_argument('--seed', type=int,
                       help='Random number generator seed.')
    add_overwrite_arg(out_g)

    out_g.add_argument('--save_seeds', action='store_true',
                       help='If set, save the seeds used for the tracking \n '
                            'in the data_per_streamline property.')

    log_g = p.add_argument_group('Logging options')
    add_verbose_arg(log_g)

    return p


def _get_direction_getter(args):
    sh_data = nib.load(args.in_sh).get_fdata(dtype=np.float32)
    sphere = HemiSphere.from_sphere(get_sphere(args.sphere))
    theta = get_theta(args.theta, args.algo)

    non_zeros_count = np.count_nonzero(np.sum(sh_data, axis=-1))
    non_first_val_count = np.count_nonzero(np.argmax(sh_data, axis=-1))

    if args.algo in ['det', 'prob']:
        if non_first_val_count / non_zeros_count > 0.5:
            logging.warning('Input detected as peaks. Input should be'
                            'fodf for det/prob, verify input just in case.')
        if args.algo == 'det':
            dg_class = DeterministicMaximumDirectionGetter
        else:
            dg_class = ProbabilisticDirectionGetter
        return dg_class.from_shcoeff(
            shcoeff=sh_data, max_angle=theta, sphere=sphere,
            basis_type=args.sh_basis,
            relative_peak_threshold=args.sf_threshold)
    elif args.algo == 'eudx':
        # Code for type EUDX. We don't use peaks_from_model
        # because we want the peaks from the provided sh.
        sh_shape_3d = sh_data.shape[:-1]
        dg = PeaksAndMetrics()
        dg.sphere = sphere
        dg.ang_thr = theta
        dg.qa_thr = args.sf_threshold

        # Heuristic to find out if the input are peaks or fodf
        # fodf are always around 0.15 and peaks around 0.75
        if non_first_val_count / non_zeros_count > 0.5:
            logging.info('Input detected as peaks.')
            nb_peaks = sh_data.shape[-1] // 3
            slices = np.arange(0, 15+1, 3)
            peak_values = np.zeros(sh_shape_3d+(nb_peaks,))
            peak_indices = np.zeros(sh_shape_3d+(nb_peaks,))

            for idx in np.argwhere(np.sum(sh_data, axis=-1)):
                idx = tuple(idx)
                for i in range(nb_peaks):
                    peak_values[idx][i] = np.linalg.norm(
                        sh_data[idx][slices[i]:slices[i+1]], axis=-1)
                    peak_indices[idx][i] = sphere.find_closest(
                        sh_data[idx][slices[i]:slices[i+1]])

            dg.peak_dirs = sh_data
        else:
            logging.info('Input detected as fodf.')
            npeaks = 5
            peak_dirs = np.zeros((sh_shape_3d + (npeaks, 3)))
            peak_values = np.zeros((sh_shape_3d + (npeaks, )))
            peak_indices = np.full((sh_shape_3d + (npeaks, )), -1, dtype='int')
            b_matrix = get_b_matrix(
                find_order_from_nb_coeff(sh_data), sphere, args.sh_basis)

            for idx in np.argwhere(np.sum(sh_data, axis=-1)):
                idx = tuple(idx)
                directions, values, indices = get_maximas(sh_data[idx],
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


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    verify_tracking_args(parser, args)
    mask_data, voxel_size = load_mask_and_verify_anisotropy(args.in_mask,
                                                            args.in_sh)
    seed_img, seeds = prepare_seeds(args.in_seed, args.random_seed,
                                    npv=args.npv, nt=args.nt)

    # Tracking is performed in voxel space
    vox_step_size = args.step_size / voxel_size
    max_steps = int(args.max_length / args.step_size) + 1
    streamlines = LocalTracking(
        _get_direction_getter(args),
        BinaryStoppingCriterion(mask_data),
        seeds, np.eye(4),
        step_size=vox_step_size, max_cross=1,
        maxlen=max_steps,
        fixedstep=True, return_all=True,
        random_seed=args.seed,
        save_seeds=args.save_seeds)

    # Using seed_img as ref to save results
    save_results(streamlines, voxel_size, seed_img,
                 args.min_length, args.max_length, args.save_seeds,
                 args.compress, args.out_tractogram)


if __name__ == '__main__':
    main()
