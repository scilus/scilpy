#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute Constrained Spherical Deconvolution (CSD) fiber ODFs.

By default, will output all possible files, using default names. Specific names
can be specified using the file flags specified in the "File flags" section.

If --not_all is set, only the files specified explicitly by the flags
will be output.

See [Tournier et al. NeuroImage 2007] and [Cote et al Tractometer MedIA 2013]
for quantitative comparisons with Sharpening Deconvolution Transform (SDT)
"""

from __future__ import division

import argparse
import logging

from dipy.io.gradients import read_bvals_bvecs
from dipy.direction.peaks import reshape_peaks_for_visualization
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_force_b0_arg,
                             add_sh_basis_args)
from scilpy.utils.bvec_bval_tools import check_b0_threshold
from scilpy.dwi.compute_fodf import compute_fodf


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Path of the input diffusion volume.')
    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')
    p.add_argument('frf_file',
                   help='Path of the FRF file')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order used for the CSD. (Default: 8)')
    p.add_argument(
        '--mask', metavar='',
        help='Path to a binary mask. Only the data inside the mask will be '
             'used for computations and reconstruction.')
    p.add_argument(
        '--processes', dest='nbr_processes', metavar='NBR', type=int,
        help='Number of sub processes to start. Default : cpu count')

    p.add_argument(
        '--not_all', action='store_true',
        help='If set, only saves the files specified using the file flags. '
             '(Default: False)')

    add_force_b0_arg(p)
    add_sh_basis_args(p)

    g = p.add_argument_group(title='File flags')

    g.add_argument(
        '--fodf', metavar='file', default='',
        help='Output filename for the fiber ODF coefficients.')
    g.add_argument(
        '--peaks', metavar='file', default='',
        help='Output filename for the extracted peaks.')
    g.add_argument(
        '--peak_indices', metavar='file', default='',
        help='Output filename for the generated peaks indices on the sphere.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.not_all:
        args.fodf = args.fodf or 'fodf.nii.gz'
        args.peaks = args.peaks or 'peaks.nii.gz'
        args.peak_indices = args.peak_indices or 'peak_indices.nii.gz'

    arglist = [args.fodf, args.peaks, args.peak_indices]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least '
                     'one file to output.')

    assert_inputs_exist(parser, [args.input, args.bvals, args.bvecs,
                                 args.frf_file])
    assert_outputs_exist(parser, args, arglist)

    full_frf = np.loadtxt(args.frf_file)

    vol = nib.load(args.input)
    data = vol.get_data()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    check_b0_threshold(args, bvals.min())

    if args.mask is None:
        mask = None
    else:
        mask = nib.load(args.mask).get_data().astype(np.bool)

    peaks_csd = compute_fodf(data, bvals, bvecs, full_frf,
                             sh_order=args.sh_order,
                             nbr_processes=args.nbr_processes,
                             mask=mask, sh_basis=args.sh_basis,
                             return_sh=True)

    if args.fodf:
        nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype(np.float32),
                                 vol.affine), args.fodf)

    if args.peaks:
        nib.save(nib.Nifti1Image(
            reshape_peaks_for_visualization(peaks_csd), vol.affine),
            args.peaks)

    if args.peak_indices:
        nib.save(nib.Nifti1Image(peaks_csd.peak_indices, vol.affine),
                 args.peak_indices)


if __name__ == "__main__":
    main()
