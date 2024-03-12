#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to denoise a dataset with the Non Local Means algorithm.

Formerly: scil_run_nlmeans.py
"""

import argparse
import logging
import warnings

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_processes_arg,
                             add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_headers_compatible)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_image',
                   help='Path of the image file to denoise.')
    p.add_argument('out_image',
                   help='Path to save the denoised image file.')
    p.add_argument('number_coils', type=int,
                   help='Number of receiver coils of the scanner.\nUse '
                        'number_coils=1 in the case of a SENSE (GE, Philips) '
                        'reconstruction and \nnumber_coils >= 1 for GRAPPA '
                        'reconstruction (Siemens). number_coils=4 works well '
                        'for the 1.5T\n in Sherbrooke. Use number_coils=0 if '
                        'the noise is considered Gaussian distributed.')

    p.add_argument('--mask', metavar='',
                   help='Path to a binary mask. Only the data inside the mask'
                        ' will be used for computations')
    p.add_argument('--sigma', metavar='float', type=float,
                   help='The standard deviation of the noise to use instead '
                        'of computing  it automatically.')
    p.add_argument('--log', dest="logfile",
                   help='If supplied, name of the text file to store '
                        'the logs.')

    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def _get_basic_sigma(data):
    # We force to zero as the 3T is either oversmoothed or still noisy, but
    # we prefer the second option
    logging.info("In basic noise estimation, number_coils=0 is enforced!")
    sigma = estimate_sigma(data, N=0)

    # Use a single value for all of the volumes.
    # This is the same value for a given bval with this estimator
    sigma = np.median(sigma)
    logging.info('The noise standard deviation from the basic estimation '
                 'is {}'.format(sigma))

    # Broadcast the single value to a whole 3D volume for nlmeans
    return np.ones(data.shape[:3]) * sigma


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image, args.mask)
    assert_outputs_exist(parser, args, args.out_image, args.logfile)
    assert_headers_compatible(parser, args.in_image, args.mask)

    if args.logfile is not None:
        logging.getLogger().addHandler(logging.FileHandler(args.logfile,
                                                           mode='w'))

    vol = nib.load(args.in_image)
    data = vol.get_fdata(dtype=np.float32)
    if args.mask is None:
        mask = np.zeros(data.shape[0:3], dtype=bool)
        if data.ndim == 4:
            mask[np.sum(data, axis=-1) > 0] = 1
        else:
            mask[data > 0] = 1
    else:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)

    sigma = args.sigma

    if sigma is not None:
        logging.info('User supplied noise standard deviation is '
                     '{}'.format(sigma))
        # Broadcast the single value to a whole 3D volume for nlmeans
        sigma = np.ones(data.shape[:3]) * sigma
    else:
        logging.info('Estimating noise')
        sigma = _get_basic_sigma(vol.get_fdata(dtype=np.float32))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        data_denoised = nlmeans(
            data, sigma, mask=mask, rician=args.number_coils > 0,
            num_threads=args.nbr_processes)

    nib.save(nib.Nifti1Image(data_denoised, vol.affine, header=vol.header),
             args.out_image)


if __name__ == "__main__":
    main()
