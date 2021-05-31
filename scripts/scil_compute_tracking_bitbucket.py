#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local streamline HARDI tractography. The tracking is done
inside a binary mask. Streamlines greater than minL and shorter
than maxL are outputted. The tracking direction is chosen in the
aperture cone defined by the previous tracking direction and the
angular constraint. The relation between theta and the curvature
is theta=2*arcsin(step_size/(2*R)).

Algo 'det': the maxima of the spherical function (SF) the most closely
aligned to the previous direction.

Algo 'prob': a direction drawn from the empirical distribution function
defined from the SF. Default parameters as in [1].

References: [1] Girard, G., Whittingstall K., Deriche, R., and
            Descoteaux, M. (2014). Towards quantitative connectivity analysis:
            reducing tractography biases. Neuroimage, 98, 266-278.
"""

from __future__ import division

import argparse
import logging
import math
import os
import time

import dipy.core.geometry as gm
import nibabel as nib
import numpy as np

from dipy.tracking.streamlinespeed import compress_streamlines
from scilpy.io.utils import (add_sh_basis_args, add_overwrite_arg,
                             add_verbose_arg)
from scilpy.io.streamlines import save_streamlines
from scilpy.tracking.trackable_dataset import Dataset, Seed, BinaryMask
from scilpy.tracking.local_tracking_bitbucket import track
from scilpy.tracking.tracker import (probabilisticTracker,
                                     deterministicMaximaTracker)
from scilpy.tracking.tracking_field import SphericalHarmonicField


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p._optionals.title = 'Generic options'
    p.add_argument('in_sh',
                   help='Spherical harmonic file (.nii.gz).')
    p.add_argument('in_seed',
                   help='Seeding mask  (.nii.gz).')
    p.add_argument('in_mask',
                   help='Seeding mask (.nii.gz).\n'
                        'Tracking will stop outside this mask.')
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')

    add_sh_basis_args(p)
    p.add_argument('--algo', default='det', choices=['det', 'prob'],
                   help='Algorithm to use (must be \'det\' or \'prob\'). '
                        '[%(default)s]')

    seeding_group = p.add_mutually_exclusive_group()
    seeding_group.add_argument('--npv', metavar='NBR', type=int,
                               help='Number of seeds per voxel. [1]')
    seeding_group.add_argument('--nt', metavar='NBR', type=int,
                               help='Total number of seeds. Replaces --npv '
                                    'and --ns.')
    seeding_group.add_argument('--ns', metavar='NBR', type=int,
                               help='Number of streamlines to estimate. ' +
                                    'Replaces --npv and\n--nt. No ' +
                                    'multiprocessing is used.')

    p.add_argument('--skip', metavar='NBR', type=int, default=0,
                   help='Skip the first NBR generated seeds / NBR seeds per' +
                        ' voxel\n(--nt / --npv). Not working with --ns. ' +
                        '[%(default)s]')
    p.add_argument('--random', type=int, default=0,
                   help='Initial value for the random number generator. ' +
                        '[%(default)s]')
    p.add_argument('--step', dest='step_size', type=float, default=0.5,
                   help='Step size in mm. [%(default)s]')
    p.add_argument('--rk_order', type=int, default=2, choices=[1, 2, 4],
                   help='The order of the Runge-Kutta integration used for\n' +
                        'the step function [%(default)s]\n' +
                        'As a rule of thumb, doubling the rk_order will \n' +
                        'double the computation time in the worst case.')
    p.add_argument('--theta', metavar='ANGLE', type=float,
                   help='Maximum angle (in degrees) between 2 steps. \n' +
                        '[\'det\'=45, \'prob\'=20]')
    p.add_argument('--maxL_no_dir', metavar='MAX', type=float, default=1,
                   help='Maximum length without valid direction, in mm. ' +
                        '[%(default)s]')
    p.add_argument('--sfthres', dest='sf_threshold', metavar='THRES',
                   type=float, default=0.1,
                   help='Spherical function relative threshold. [%(default)s]')
    p.add_argument('--sfthres_init', dest='sf_threshold_init',
                   metavar='THRES', type=float, default=0.5,
                   help='Spherical function relative threshold value\n' +
                        'for the initial direction. [%(default)s]')
    p.add_argument('--minL', dest='min_length', type=float, default=10,
                   help='Minimum length of a streamline in mm. [%(default)s]')
    p.add_argument('--maxL', dest='max_length', type=int, default=300,
                   help='Maximum length of a streamline in mm. [%(default)s]')

    p.add_argument('--sh_interp', dest='field_interp',
                   default='tl', choices=['nn', 'tl'],
                   help="Spherical harmonic interpolation: \n'nn' " +
                        "(nearest-neighbor) or 'tl' (trilinear). " +
                        "[%(default)s]")
    p.add_argument('--mask_interp', dest='mask_interp',
                   default='nn', choices=['nn', 'tl'],
                   help="Mask interpolation:\n'nn' (nearest-neighbor) or " +
                        "'tl' (trilinear). [%(default)s]")

    p.add_argument('--single_direction', dest='is_single_direction',
                   action='store_true',
                   help="If set, tracks in one direction only (forward or \n" +
                        "backward) given the initial seed. The direction is" +
                        "\nrandomly drawn from the ODF.")
    p.add_argument('--processes', dest='nbr_processes', type=int, default=0,
                   help='Number of sub processes to start. [cpu count]')
    p.add_argument('--load_data', action='store_true', dest='isLoadData',
                   help='If set, loads data in memory for all processes. \n' +
                        'Increases the speed, and the memory requirements.')
    p.add_argument('--compress', type=float,
                   help='If set, will compress streamlines. The parameter\n' +
                        'value is the distance threshold. A rule of thumb\n ' +
                        'is to set it to 0.1mm for deterministic\n' +
                        'streamlines and 0.2mm for probabilitic streamlines.')
    p.add_argument('--save_seeds', action='store_true',
                   help='If set, each streamline generated will save \n' +
                        'its 3D seed point in the TRK file using `seed` in' +
                        ' \nthe \'data_per_streamline\' attribute')
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    param = {}

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    if not np.any([args.nt, args.npv, args.ns]):
        args.npv = 1

    if not args.min_length > 0:
        parser.error('minL must be > 0, {0}mm was provided.'
                     .format(args.min_length))
    if args.max_length < args.min_length:
        parser.error('maxL must be > than minL, (minL={0}mm, maxL={1}mm).'
                     .format(args.min_length, args.max_length))

    if args.theta is not None:
        theta = gm.math.radians(args.theta)
    elif args.algo == 'prob':
        theta = gm.math.radians(20)
    else:
        theta = gm.math.radians(45)

    if args.mask_interp == 'nn':
        mask_interpolation = 'nearest'
    elif args.mask_interp == 'tl':
        mask_interpolation = 'trilinear'
    else:
        parser.error("--mask_interp has wrong value. See the help (-h).")
        return

    if args.field_interp == 'nn':
        field_interpolation = 'nearest'
    elif args.field_interp == 'tl':
        field_interpolation = 'trilinear'
    else:
        parser.error("--sh_interp has wrong value. See the help (-h).")
        return

    param['random'] = args.random
    param['skip'] = args.skip
    param['algo'] = args.algo
    param['mask_interp'] = mask_interpolation
    param['field_interp'] = field_interpolation
    param['theta'] = theta
    param['sf_threshold'] = args.sf_threshold
    param['sf_threshold_init'] = args.sf_threshold_init
    param['step_size'] = args.step_size
    param['rk_order'] = args.rk_order
    param['max_length'] = args.max_length
    param['min_length'] = args.min_length
    param['max_nbr_pts'] = int(param['max_length'] / param['step_size'])
    param['min_nbr_pts'] = int(param['min_length'] / param['step_size']) + 1
    param['is_single_direction'] = args.is_single_direction
    param['nbr_seeds'] = args.nt if args.nt is not None else 0
    param['nbr_seeds_voxel'] = args.npv if args.npv is not None else 0
    param['nbr_streamlines'] = args.ns if args.ns is not None else 0
    param['max_no_dir'] = int(math.ceil(args.maxL_no_dir / param['step_size']))
    param['is_all'] = False
    param['is_keep_single_pts'] = False
    # r+ is necessary for interpolation function in cython who
    # need read/write right
    param['mmap_mode'] = None if args.isLoadData else 'r+'
    logging.debug('Tractography parameters:\n{0}'.format(param))

    seed_img = nib.load(args.in_seed)
    seed = Seed(seed_img)
    if args.npv:
        param['nbr_seeds'] = len(seed.seeds) * param['nbr_seeds_voxel']
        param['skip'] = len(seed.seeds) * param['skip']
    if len(seed.seeds) == 0:
        parser.error('"{0}" does not have voxels value > 0.'
                     .format(args.in_seed))

    mask = BinaryMask(
        Dataset(nib.load(args.in_mask), param['mask_interp']))

    dataset = Dataset(nib.load(args.in_sh), param['field_interp'])
    field = SphericalHarmonicField(dataset,
                                   args.sh_basis,
                                   param['sf_threshold'],
                                   param['sf_threshold_init'],
                                   param['theta'])

    if args.algo == 'det':
        tracker =\
            deterministicMaximaTracker(field, args.step_size, args.rk_order)
    elif args.algo == 'prob':
        tracker = probabilisticTracker(field, args.step_size, args.rk_order)
    else:
        parser.error("--algo has wrong value. See the help (-h).")
        return

    start = time.time()
    if args.compress:
        if args.compress < 0.001 or args.compress > 1:
            logging.warn('You are using an error rate of {}.\n'
                         .format(args.compress) +
                         'We recommend setting it between 0.001 and 1.\n' +
                         '0.001 will do almost nothing to the tracts while ' +
                         '1 will higly compress/linearize the tracts')

        streamlines, seeds = track(tracker, mask, seed, param, compress=True,
                                   compression_error_threshold=args.compress,
                                   nbr_processes=args.nbr_processes,
                                   pft_tracker=None,
                                   save_seeds=args.save_seeds)
    else:
        streamlines, seeds = track(tracker, mask, seed, param,
                                   nbr_processes=args.nbr_processes,
                                   pft_tracker=None,
                                   save_seeds=args.save_seeds)

    if args.compress:
        streamlines = (
            compress_streamlines(s, args.compress)
            for s in streamlines)

    save_streamlines(streamlines, args.in_seed, args.out_tractogram, seeds)

    str_time = "%.2f" % (time.time() - start)
    logging.debug(str(len(streamlines)) + " streamlines, done in " +
                  str_time + " seconds.")


if __name__ == "__main__":
    main()
