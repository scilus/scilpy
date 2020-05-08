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

import argparse
import logging

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.direction.peaks import reshape_peaks_for_visualization
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_force_b0_arg,
                             add_sh_basis_args, add_processes_arg)
from scilpy.reconst.multi_processes import (fit_from_model, peaks_from_sh,
                                            convert_sh_basis)
from scilpy.utils.bvec_bval_tools import (check_b0_threshold, normalize_bvecs,
                                          is_normalized_bvecs)


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
        '--not_all', action='store_true',
        help='If set, only saves the files specified using the file flags. '
             '(Default: False)')

    add_force_b0_arg(p)
    add_sh_basis_args(p)
    add_processes_arg(p)

    g = p.add_argument_group(title='File flags')

    g.add_argument(
        '--fodf', metavar='file', default='',
        help='Output filename for the fiber ODF coefficients.')
    g.add_argument(
        '--peaks', metavar='file', default='',
        help='Output filename for the extracted peaks.')
    g.add_argument(
        '--peak_values', metavar='file', default='',
        help='Output filename for the extracted peaks values.')
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
        args.peak_values = args.peak_values or 'peak_values.nii.gz'
        args.peak_indices = args.peak_indices or 'peak_indices.nii.gz'

    arglist = [args.fodf, args.peaks, args.peak_values, args.peak_indices]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least '
                     'one file to output.')

    assert_inputs_exist(parser, [args.input, args.bvals, args.bvecs,
                                 args.frf_file])
    assert_outputs_exist(parser, args, arglist)

    # Loading data
    full_frf = np.loadtxt(args.frf_file)
    vol = nib.load(args.input)
    data = vol.get_fdata(dtype=np.float32)
    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    # Checking mask
    if args.mask is None:
        mask = None
    else:
        mask = np.asanyarray(nib.load(args.mask).dataobj).astype(np.bool)
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")

    sh_order = args.sh_order

    # Checking data and sh_order
    check_b0_threshold(args.force_b0_threshold, bvals.min())
    if data.shape[-1] < (sh_order + 1) * (sh_order + 2) / 2:
        logging.warning(
            'We recommend having at least {} unique DWI volumes, but you '
            'currently have {} volumes. Try lowering the parameter sh_order '
            'in case of non convergence.'.format(
                (sh_order + 1) * (sh_order + 2) / 2, data.shape[-1]))

    # Checking bvals, bvecs values and loading gtab
    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    # Checking full_frf and separating it
    if not full_frf.shape[0] == 4:
        raise ValueError('FRF file did not contain 4 elements. '
                         'Invalid or deprecated FRF format')
    frf = full_frf[0:3]
    mean_b0_val = full_frf[3]

    # Loading the spheres
    reg_sphere = get_sphere('symmetric362')
    peaks_sphere = get_sphere('symmetric724')

    # Computing CSD
    csd_model = ConstrainedSphericalDeconvModel(
        gtab, (frf, mean_b0_val),
        reg_sphere=reg_sphere,
        sh_order=sh_order)

    # Computing CSD fit
    csd_fit = fit_from_model(csd_model, data,
                             mask=mask, nbr_processes=args.nbr_processes)

    if args.peaks or args.peak_values or args.peak_indices:
        # Computing peaks
        peak_dirs, peak_values, \
            peak_indices = peaks_from_sh(csd_fit.shm_coeff,
                                         peaks_sphere,
                                         mask=mask,
                                         relative_peak_threshold=.5,
                                         min_separation_angle=25,
                                         normalize_peaks=True,
                                         nbr_processes=args.nbr_processes)

    # Saving results
    if args.fodf:
        shm_coeff = csd_fit.shm_coeff
        if args.sh_basis == 'tournier07':
            shm_coeff = convert_sh_basis(shm_coeff, peaks_sphere, mask=mask,
                                         nbr_processes=args.nbr_processes)
        nib.save(nib.Nifti1Image(shm_coeff.astype(np.float32),
                                 vol.affine), args.fodf)

    if args.peaks:
        nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peak_dirs),
                                 vol.affine), args.peaks)

    if args.peak_values:
        nib.save(nib.Nifti1Image(peak_values, vol.affine), args.peak_values)

    if args.peak_indices:
        nib.save(nib.Nifti1Image(peak_indices, vol.affine), args.peak_indices)


if __name__ == "__main__":
    main()
