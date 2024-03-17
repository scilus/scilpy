#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute powder average (mean diffusion weighted image) from set of
diffusion images.

By default will output an average image calculated from all images with
non-zero bvalue.

Specify --bvalue to output an image for a single shell

Script currently does not take into account the diffusion gradient directions
being averaged.

Formerly: scil_compute_powder_average.py
"""

import argparse
import logging

import nibabel as nib

import numpy as np

from dipy.core.gradients import get_bval_indices
from dipy.io.gradients import read_bvals_bvecs

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg,
                             assert_headers_compatible)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='Path of the input diffusion volume.')
    p.add_argument('in_bval',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('out_avg',
                   help='Path of the output file.')

    add_overwrite_arg(p)

    p.add_argument('--mask', metavar='file',
                   help='Path to a binary mask.\nOnly data inside the'
                   ' mask will be used for powder avg. '
                   '(Default: %(default)s)')

    p.add_argument('--b0_thr', type=int, default='50',
                   help='Exclude b0 volumes from powder average with'
                   ' bvalue less than specified threshold.\n'
                   '(Default: remove volumes with bvalue < %(default)s')

    p.add_argument('--shells', nargs='+', type=int, default=None,
                   help='bvalue (shells) to include in powder average'
                   ' passed as a list \n(e.g. --shells 1000 2000). '
                   'If not specified will include all volumes with'
                   ' a non-zero bvalue.')

    p.add_argument('--shell_thr', type=int, default='50',
                   help='Include volumes with bvalue +- the specified'
                   ' threshold.\n(Default: [%(default)s]')

    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval], args.mask)
    assert_outputs_exist(parser, args, args.out_avg)
    assert_headers_compatible(parser, args.in_dwi, args.mask)

    img = nib.load(args.in_dwi)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    mask = get_data_as_mask(nib.load(args.mask),
                            dtype='uint8') if args.mask else None

    # Read bvals (bvecs not needed at this point)
    logging.info('Performing powder average')
    bvals, _ = read_bvals_bvecs(args.in_bval, None)

    # Select diffusion volumes to average
    if not (args.shells):
        # If no shell given, average all diffusion weighted images
        pwd_avg_idx = np.squeeze(np.where(bvals > 0 + args.b0_thr))
        logging.debug('Calculating powder average from all diffusion'
                      '-weighted volumes, {} volumes '
                      'included.'.format(len(pwd_avg_idx)))
    else:
        pwd_avg_idx = []
        logging.debug('Calculating powder average from {} '
                      'shells {}'.format(len(args.shells), args.shells))

        for shell in args.shells:
            pwd_avg_idx = np.int64(
                np.concatenate((pwd_avg_idx,
                                get_bval_indices(bvals,
                                                 shell,
                                                 tol=args.shell_thr))))
            logging.debug('{} b{} volumes detected and included'.format(
                len(pwd_avg_idx), shell))

        # remove b0 indices
        b0_idx = get_bval_indices(bvals, 0, args.b0_thr)
        logging.debug('{} b0 volumes detected and not included'.format(
            len(b0_idx)))
        for val in b0_idx:
            pwd_avg_idx = pwd_avg_idx[pwd_avg_idx != val]

    if len(pwd_avg_idx) == 0:
        raise ValueError('No shells selected for powder average, ensure '
                         'shell, shell_thr and b0_thr are set '
                         'appropriately')

    powder_avg = np.squeeze(np.mean(data[:, :, :, pwd_avg_idx], axis=3))

    if args.mask:
        powder_avg = powder_avg * mask

    powder_avg_img = nib.Nifti1Image(powder_avg.astype(np.float32), affine)
    nib.save(powder_avg_img, args.out_avg)


if __name__ == "__main__":
    main()
