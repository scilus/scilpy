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
"""

import argparse
import logging

from dipy.core.sphere import HemiSphere
from dipy.data import SPHERE_FILES, get_sphere
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
from nibabel.streamlines.tractogram import LazyTractogram
import numpy as np

from scilpy.reconst.utils import (find_order_from_nb_coeff,
                                  get_b_matrix, get_maximas)
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.tracking.tools import get_theta


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p._optionals.title = 'Generic options'
    p.add_argument('in_sh',
                   help='Spherical harmonic file. \n'
                        '(isotropic resolution, nifti, see --basis).')
    p.add_argument('in_seed',
                   help='Seeding mask (isotropic resolution, nifti).')
    p.add_argument('in_mask',
                   help='Seeding mask(isotropic resolution, nifti).\n' +
                        'Tracking will stop outside this mask.')
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be trk or tck).')

    track_g = p.add_argument_group('Tracking options')
    track_g.add_argument(
        '--algo', default='prob', choices=['det', 'prob'],
        help='Algorithm to use (must be "det" or "prob"). [%(default)s]')
    track_g.add_argument(
        '--step', dest='step_size', type=float, default=0.5,
        help='Step size in mm. [%(default)s]')
    track_g.add_argument(
        '--min_length', type=float, default=10.,
        help='Minimum length of a streamline in mm. [%(default)s]')
    track_g.add_argument(
        '--max_length', type=float, default=300.,
        help='Maximum length of a streamline in mm. [%(default)s]')
    track_g.add_argument(
        '--theta', type=float,
        help='Maximum angle between 2 steps. ["eudx"=60, det"=45, "prob"=20]')
    track_g.add_argument(
        '--sfthres', dest='sf_threshold', type=float, default=0.1,
        help='Spherical function relative threshold. [%(default)s]')
    add_sh_basis_args(track_g)

    seed_group = p.add_argument_group(
        'Seeding options',
        'When no option is provided, uses --npv 1.')
    seed_sub_exclusive = seed_group.add_mutually_exclusive_group()
    seed_sub_exclusive.add_argument(
        '--npv', type=int,
        help='Number of seeds per voxel.')
    seed_sub_exclusive.add_argument(
        '--nt', type=int,
        help='Total number of seeds to use.')

    p.add_argument(
        '--sphere', choices=sorted(SPHERE_FILES.keys()),
        default='symmetric724',
        help='Set of directions to be used for tracking.')

    out_g = p.add_argument_group('Output options')
    out_g.add_argument(
        '--compress', type=float,
        help='If set, will compress streamlines. The parameter\nvalue is the '
             'distance threshold. A rule of thumb\nis to set it to 0.1mm for '
             'deterministic\nstreamlines and 0.2mm for probabilitic '
             'streamlines.')
    out_g.add_argument(
        '--seed', type=int,
        help='Random number generator seed.')
    add_overwrite_arg(out_g)

    out_g.add_argument(
        '--save_seeds', action='store_true',
        help='If set, save the seeds used for the tracking in the '
             'data_per_streamline property of the tractogram.')

    log_g = p.add_argument_group('Logging options')
    add_verbose_arg(log_g)

    return p


def _get_direction_getter(args, mask_data):
    sh_data = nib.load(args.in_sh).get_fdata(dtype=np.float32)
    sphere = HemiSphere.from_sphere(get_sphere(args.sphere))
    theta = get_theta(args.theta, args.algo)

    if args.algo in ['det', 'prob']:
        if args.algo == 'det':
            dg_class = DeterministicMaximumDirectionGetter
        else:
            dg_class = ProbabilisticDirectionGetter
        return dg_class.from_shcoeff(
            shcoeff=sh_data, max_angle=theta, sphere=sphere,
            basis_type=args.sh_basis,
            relative_peak_threshold=args.sf_threshold)

    # Code for type EUDX. We don't use peaks_from_model
    # because we want the peaks from the provided sh.
    sh_shape_3d = sh_data.shape[:-1]
    npeaks = 5
    peak_dirs = np.zeros((sh_shape_3d + (npeaks, 3)))
    peak_values = np.zeros((sh_shape_3d + (npeaks, )))
    peak_indices = np.full((sh_shape_3d + (npeaks, )), -1, dtype='int')
    b_matrix = get_b_matrix(
        find_order_from_nb_coeff(sh_data), sphere, args.sh_basis)

    for idx in np.ndindex(sh_shape_3d):
        if not mask_data[idx]:
            continue

        directions, values, indices = get_maximas(
            sh_data[idx], sphere, b_matrix, args.sf_threshold, 0)
        if values.shape[0] != 0:
            n = min(npeaks, values.shape[0])
            peak_dirs[idx][:n] = directions[:n]
            peak_values[idx][:n] = values[:n]
            peak_indices[idx][:n] = indices[:n]

    dg = PeaksAndMetrics()
    dg.sphere = sphere
    dg.peak_dirs = peak_dirs
    dg.peak_values = peak_values
    dg.peak_indices = peak_indices
    dg.ang_thr = theta
    dg.qa_thr = args.sf_threshold
    return dg


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    assert_inputs_exist(parser, [args.in_sh, args.in_seed, args.in_mask])
    assert_outputs_exist(parser, args, args.out_tractogram)

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    if not args.min_length > 0:
        parser.error('minL must be > 0, {}mm was provided.'
                     .format(args.min_length))
    if args.max_length < args.min_length:
        parser.error('maxL must be > than minL, (minL={}mm, maxL={}mm).'
                     .format(args.min_length, args.max_length))

    if args.compress:
        if args.compress < 0.001 or args.compress > 1:
            logging.warning(
                'You are using an error rate of {}.\nWe recommend setting it '
                'between 0.001 and 1.\n0.001 will do almost nothing to the '
                'tracts while 1 will higly compress/linearize the tracts'
                .format(args.compress))

    if args.npv and args.npv <= 0:
        parser.error('Number of seeds per voxel must be > 0.')

    if args.nt and args.nt <= 0:
        parser.error('Total number of seeds must be > 0.')

    mask_img = nib.load(args.in_mask)
    mask_data = get_data_as_mask(mask_img, dtype=np.bool)

    # Make sure the mask is isotropic. Else, the strategy used
    # when providing information to dipy (i.e. working as if in voxel space)
    # will not yield correct results.
    fodf_sh_img = nib.load(args.in_sh)
    if not np.allclose(np.mean(fodf_sh_img.header.get_zooms()[:3]),
                       fodf_sh_img.header.get_zooms()[0], atol=1.e-3):
        parser.error(
            'SH file is not isotropic. Tracking cannot be ran robustly.')

    if args.npv:
        nb_seeds = args.npv
        seed_per_vox = True
    elif args.nt:
        nb_seeds = args.nt
        seed_per_vox = False
    else:
        nb_seeds = 1
        seed_per_vox = True

    voxel_size = fodf_sh_img.header.get_zooms()[0]
    vox_step_size = args.step_size / voxel_size
    seed_img = nib.load(args.in_seed)
    seeds = track_utils.random_seeds_from_mask(
        seed_img.get_fdata(dtype=np.float32),
        np.eye(4),
        seeds_count=nb_seeds,
        seed_count_per_voxel=seed_per_vox,
        random_seed=args.seed)

    # Tracking is performed in voxel space
    max_steps = int(args.max_length / args.step_size) + 1
    streamlines = LocalTracking(
        _get_direction_getter(args, mask_data),
        BinaryStoppingCriterion(mask_data),
        seeds, np.eye(4),
        step_size=vox_step_size, max_cross=1,
        maxlen=max_steps,
        fixedstep=True, return_all=True,
        random_seed=args.seed,
        save_seeds=args.save_seeds)

    scaled_min_length = args.min_length / voxel_size
    scaled_max_length = args.max_length / voxel_size

    if args.save_seeds:
        filtered_streamlines, seeds = \
            zip(*((s, p) for s, p in streamlines
                  if scaled_min_length <= length(s) <= scaled_max_length))
        data_per_streamlines = {'seeds': lambda: seeds}
    else:
        filtered_streamlines = \
            (s for s in streamlines
             if scaled_min_length <= length(s) <= scaled_max_length)
        data_per_streamlines = {}

    if args.compress:
        filtered_streamlines = (
            compress_streamlines(s, args.compress)
            for s in filtered_streamlines)

    tractogram = LazyTractogram(lambda: filtered_streamlines,
                                data_per_streamlines,
                                affine_to_rasmm=seed_img.affine)

    filetype = nib.streamlines.detect_format(args.out_tractogram)
    reference = get_reference_info(seed_img)
    header = create_tractogram_header(filetype, *reference)

    # Use generator to save the streamlines on-the-fly
    nib.streamlines.save(tractogram, args.out_tractogram, header=header)


if __name__ == '__main__':
    main()
