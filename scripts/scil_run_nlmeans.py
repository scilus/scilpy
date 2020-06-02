#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to denoise a dataset with the Non Local Means algorithm.
"""

import argparse
import logging
import warnings

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
import nibabel as nb
import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'input',
        help='Path of the image file to denoise.')
    p.add_argument(
        'output',
        help='Path to save the denoised image file.')
    p.add_argument(
        'N', metavar='number_coils', type=int,
        help='Number of receiver coils of the scanner.\nUse N=1 in the case '
             'of a SENSE (GE, Philips) reconstruction and \nN >= 1 for '
             'GRAPPA reconstruction (Siemens). N=4 works well for the 1.5T\n'
             'in Sherbrooke. Use N=0 if the noise is considered Gaussian '
             'distributed.')

    p.add_argument(
        '--mask', metavar='',
        help='Path to a binary mask. Only the data inside the mask will be '
             'used for computations')
    p.add_argument(
        '--sigma', metavar='float', type=float,
        help='The standard deviation of the noise to use instead of computing '
             ' it automatically.')
    p.add_argument(
        '--log', dest="logfile",
        help="If supplied, name of the text file to store the logs.")
    p.add_argument(
        '--processes', dest='nbr_processes', metavar='int', type=int,
        help='Number of sub processes to start. Default: Use all cores.')
    p.add_argument(
        '-v', '--verbose', action="store_true", dest="verbose",
        help="Print more info. Default : Print only warnings.")
    add_overwrite_arg(p)
    return p


def _get_basic_sigma(data, log):
    # We force to zero as the 3T is either oversmoothed or still noisy, but
    # we prefer the second option
    log.info("In basic noise estimation, N=0 is enforced!")
    sigma = estimate_sigma(data, N=0)

    # Use a single value for all of the volumes.
    # This is the same value for a given bval with this estimator
    sigma = np.median(sigma)
    log.info('The noise standard deviation from the basic estimation '
             'is {}'.format(sigma))

    # Broadcast the single value to a whole 3D volume for nlmeans
    return np.ones(data.shape[:3]) * sigma


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.input)
    assert_outputs_exist(parser, args, args.output, args.logfile)

    logging.basicConfig()
    log = logging.getLogger(__name__)
    if args.verbose:
        log.setLevel(level=logging.INFO)
    else:
        log.setLevel(level=logging.WARNING)

    if args.logfile is not None:
        log.addHandler(logging.FileHandler(args.logfile, mode='w'))

    vol = nb.load(args.input)
    data = vol.get_data()
    if args.mask is None:
        mask = np.ones(data.shape[:3], dtype=np.bool)
    else:
        mask = nb.load(args.mask).get_data().astype(np.bool)

    sigma = args.sigma

    if sigma is not None:
        log.info('User supplied noise standard deviation is {}'.format(sigma))
        # Broadcast the single value to a whole 3D volume for nlmeans
        sigma = np.ones(data.shape[:3]) * sigma
    else:
        log.info('Estimating noise')
        sigma = _get_basic_sigma(vol.get_data(), log)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        data_denoised = nlmeans(
            data, sigma, mask=mask, rician=args.N > 0,
            num_threads=args.nbr_processes)

    nb.save(nb.Nifti1Image(
        data_denoised, vol.affine, vol.header), args.output)


if __name__ == "__main__":
    main()
