#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local streamline HARDI tractography including Particle Filtering tracking.

WARNING: This script DOES NOT support asymetric FODF input (aFODF).

The tracking is done inside partial volume estimation maps and uses the
particle filtering tractography (PFT) algorithm. See
scil_tracking_pft_maps.py to generate PFT required maps.

Streamlines longer than min_length and shorter than max_length are kept.
The tracking direction is chosen in the aperture cone defined by the
previous tracking direction and the angular constraint.
Default parameters as suggested in [1].

Algo 'det': the maxima of the spherical function (SF) the most closely aligned
to the previous direction.
Algo 'prob': a direction drawn from the empirical distribution function defined
from the SF.

For streamline compression, a rule of thumb is to set it to 0.1mm for the
deterministic algorithm and 0.2mm for probabilitic algorithm.

All the input nifti files must be in isotropic resolution.

Formerly: scil_compute_pft.py
"""

import argparse
import logging

from dipy.data import get_sphere, HemiSphere
from dipy.direction import (ProbabilisticDirectionGetter,
                            DeterministicMaximumDirectionGetter)
from dipy.io.utils import (get_reference_info,
                           create_tractogram_header)
from dipy.tracking.local_tracking import ParticleFilteringTracking
from dipy.tracking.stopping_criterion import (ActStoppingCriterion,
                                              CmcStoppingCriterion)
from dipy.tracking import utils as track_utils
from dipy.tracking.streamlinespeed import length, compress_streamlines
import nibabel as nib
from nibabel.streamlines import LazyTractogram
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, parse_sh_basis_arg,
                             assert_headers_compatible, add_compression_arg,
                             verify_compression_th)
from scilpy.tracking.utils import get_theta


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
        epilog='References: [1] Girard, G., Whittingstall K., Deriche, R., '
               'and Descoteaux, M. (2014). Towards quantitative connectivity '
               'analysis: reducing tractography biases. Neuroimage, 98, '
               '266-278.')
    p._optionals.title = 'Generic options'

    p.add_argument('in_sh',
                   help='Spherical harmonic file (.nii.gz).')
    p.add_argument('in_seed',
                   help='Seeding mask (.nii.gz).')
    p.add_argument('in_map_include',
                   help='The probability map (.nii.gz) of ending the\n'
                        'streamline and including it in the output (CMC, PFT '
                        '[1])')
    p.add_argument('map_exclude_file',
                   help='The probability map (.nii.gz) of ending the\n'
                        'streamline and excluding it in the output (CMC, PFT '
                        '[1]).')
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')

    track_g = p.add_argument_group('Tracking options')
    track_g.add_argument('--algo', default='prob', choices=['det', 'prob'],
                         help='Algorithm to use (must be "det" or "prob"). '
                              '[%(default)s]')
    track_g.add_argument('--step', dest='step_size', type=float, default=0.2,
                         help='Step size in mm. [%(default)s]')
    track_g.add_argument('--min_length', type=float, default=10.,
                         help='Minimum length of a streamline in mm. '
                              '[%(default)s]')
    track_g.add_argument('--max_length', type=float, default=300.,
                         help='Maximum length of a streamline in mm. '
                              '[%(default)s]')
    track_g.add_argument('--theta', type=float,
                         help='Maximum angle between 2 steps. '
                              '["det"=45, "prob"=20]')
    track_g.add_argument('--act', action='store_true',
                         help='If set, uses anatomically-constrained '
                              'tractography (ACT) \ninstead of continuous map '
                              'criterion (CMC).')
    track_g.add_argument('--sfthres', dest='sf_threshold',
                         type=float, default=0.1,
                         help='Spherical function relative threshold. '
                              '[%(default)s]')
    track_g.add_argument('--sfthres_init', dest='sf_threshold_init',
                         type=float, default=0.5,
                         help='Spherical function relative threshold value '
                              'for the \ninitial direction. [%(default)s]')
    add_sh_basis_args(track_g)

    seed_group = p.add_argument_group(
        'Seeding options',
        'When no option is provided, uses --npv 1.')
    seed_sub_exclusive = seed_group.add_mutually_exclusive_group()
    seed_sub_exclusive.add_argument('--npv', type=int,
                                    help='Number of seeds per voxel.')
    seed_sub_exclusive.add_argument('--nt', type=int,
                                    help='Total number of seeds to use.')

    pft_g = p.add_argument_group('PFT options')
    pft_g.add_argument('--particles', type=int, default=15,
                       help='Number of particles to use for PFT. [%(default)s]'
                       )
    pft_g.add_argument('--back', dest='back_tracking', type=float, default=2.,
                       help='Length of PFT back tracking (mm). [%(default)s]')
    pft_g.add_argument('--forward', dest='forward_tracking',
                       type=float, default=1.,
                       help='Length of PFT forward tracking (mm). '
                            '[%(default)s]')

    out_g = p.add_argument_group('Output options')
    out_g.add_argument('--all', dest='keep_all', action='store_true',
                       help='If set, keeps "excluded" streamlines.\n'
                            'NOT RECOMMENDED, except for debugging.')
    out_g.add_argument('--seed', type=int,
                       help='Random number generator seed.')
    add_overwrite_arg(out_g)

    out_g.add_argument('--save_seeds', action='store_true',
                       help='If set, save the seeds used for the tracking \n '
                            'in the data_per_streamline property.')

    add_compression_arg(out_g)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    required = [args.in_sh, args.in_seed,
                args.in_map_include, args.map_exclude_file]
    assert_inputs_exist(parser, required)
    assert_outputs_exist(parser, args, args.out_tractogram)
    assert_headers_compatible(parser, required)

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    if not args.min_length > 0:
        parser.error('minL must be > 0, {}mm was provided.'
                     .format(args.min_length))
    if args.max_length < args.min_length:
        parser.error('maxL must be > than minL, (minL={}mm, maxL={}mm).'
                     .format(args.min_length, args.max_length))

    verify_compression_th(args.compress_th)

    if args.particles <= 0:
        parser.error('--particles must be >= 1.')

    if args.back_tracking <= 0:
        parser.error('PFT backtracking distance must be > 0.')

    if args.forward_tracking <= 0:
        parser.error('PFT forward tracking distance must be > 0.')

    if args.npv and args.npv <= 0:
        parser.error('Number of seeds per voxel must be > 0.')

    if args.nt and args.nt <= 0:
        parser.error('Total number of seeds must be > 0.')

    fodf_sh_img = nib.load(args.in_sh)
    if not np.allclose(np.mean(fodf_sh_img.header.get_zooms()[:3]),
                       fodf_sh_img.header.get_zooms()[0], atol=1e-03):
        parser.error(
            'SH file is not isotropic. Tracking cannot be ran robustly.')

    tracking_sphere = HemiSphere.from_sphere(get_sphere('repulsion724'))

    # Check if sphere is unit, since we couldn't find such check in Dipy.
    if not np.allclose(np.linalg.norm(tracking_sphere.vertices, axis=1), 1.):
        raise RuntimeError('Tracking sphere should be unit normed.')

    sh_basis, is_legacy = parse_sh_basis_arg(args)

    if args.algo == 'det':
        dgklass = DeterministicMaximumDirectionGetter
    else:
        dgklass = ProbabilisticDirectionGetter

    theta = get_theta(args.theta, args.algo)

    # Reminder for the future:
    # pmf_threshold == clip pmf under this
    # relative_peak_threshold is for initial directions filtering
    # min_separation_angle is the initial separation angle for peak extraction
    dg = dgklass.from_shcoeff(
        fodf_sh_img.get_fdata(dtype=np.float32),
        max_angle=theta,
        sphere=tracking_sphere,
        basis_type=sh_basis,
        legacy=is_legacy,
        pmf_threshold=args.sf_threshold,
        relative_peak_threshold=args.sf_threshold_init)

    map_include_img = nib.load(args.in_map_include)
    map_exclude_img = nib.load(args.map_exclude_file)
    voxel_size = np.average(map_include_img.header['pixdim'][1:4])

    if not args.act:
        tissue_classifier = CmcStoppingCriterion(
            map_include_img.get_fdata(dtype=np.float32),
            map_exclude_img.get_fdata(dtype=np.float32),
            step_size=args.step_size,
            average_voxel_size=voxel_size)
    else:
        tissue_classifier = ActStoppingCriterion(
            map_include_img.get_fdata(dtype=np.float32),
            map_exclude_img.get_fdata(dtype=np.float32))

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
        get_data_as_mask(seed_img, dtype=bool),
        np.eye(4),
        seeds_count=nb_seeds,
        seed_count_per_voxel=seed_per_vox,
        random_seed=args.seed)

    # Note that max steps is used once for the forward pass, and
    # once for the backwards. This doesn't, in fact, control the real
    # max length
    max_steps = int(args.max_length / args.step_size) + 1
    pft_streamlines = ParticleFilteringTracking(
        dg,
        tissue_classifier,
        seeds,
        np.eye(4),
        max_cross=1,
        step_size=vox_step_size,
        maxlen=max_steps,
        pft_back_tracking_dist=args.back_tracking,
        pft_front_tracking_dist=args.forward_tracking,
        particle_count=args.particles,
        return_all=args.keep_all,
        random_seed=args.seed,
        save_seeds=args.save_seeds)

    scaled_min_length = args.min_length / voxel_size
    scaled_max_length = args.max_length / voxel_size

    if args.save_seeds:
        filtered_streamlines, seeds = \
            zip(*((s, p) for s, p in pft_streamlines
                  if scaled_min_length <= length(s) <= scaled_max_length))
        data_per_streamlines = {'seeds': lambda: seeds}
    else:
        filtered_streamlines = \
            (s for s in pft_streamlines
             if scaled_min_length <= length(s) <= scaled_max_length)
        data_per_streamlines = {}

    if args.compress_th:
        filtered_streamlines = (
            compress_streamlines(s, args.compress_th)
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
