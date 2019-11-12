#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute the Diffusion Kurtosis Imaging (DKI) metrics. DKI is a multi-shell
diffusion model. The input DWI needs to be multi-shell, i.e. multi-bvalued.

Since the diffusion kurtosis model involves the estimation of a large number of parameters 
and since the non-Gaussian components of the diffusion signal are more sensitive to artefacts, 
you should really denoise your DWI volume before using this DKI script (e.g. scil_run_nlmeans.py). 

By default, will output all available metrics, using default names. Specific
names can be specified using the metrics flags that are listed in the "Metrics
files flags" section.

If --not_all is set, only the metrics specified explicitly by the flags
will be output. The available metrics are:

DKI version of fractional anisotropy (FA), axial diffusivisty (AD),
radial diffusivity (RD), and mean diffusivity (MD), as well as axial kurtosis (AK), 
mean kurtosis (MK), and radial kurtosis (RK). 

This script directly comes from the DIPY example gallery and references therein.
[https://dipy.org/documentation/1.0.0./examples_built/reconst_dki/#example-reconst-dki].
"""

from __future__ import division, print_function

from builtins import range
import argparse
import logging

import nibabel as nib
import numpy as np

import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

from scipy.ndimage.filters import gaussian_filter

# Aliased to avoid clashes with images called mode.
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exists, add_force_b0_arg)
from scilpy.utils.bvec_bval_tools import (normalize_bvecs, is_normalized_bvecs,
                                          check_b0_threshold)
from scilpy.utils.bvec_bval_tools import get_shell_indices


logger = logging.getLogger("Compute_DKI_Metrics")
logger.setLevel(logging.INFO)

def _guess_bvals_centroids(bvals, threshold):
    if not len(bvals):
        raise ValueError('Empty b-values.')

    bval_centroids = [bvals[0]]

    for bval in bvals[1:]:
        diffs = np.abs(np.asarray(bval_centroids) - bval)
        if not len(np.where(diffs < threshold)[0]):
            # Found no bval in bval centroids close enough to the current one.
            bval_centroids.append(bval)

    return np.array(bval_centroids)


def identify_shells(bvals, threshold=40.0):
    centroids = _guess_bvals_centroids(bvals, threshold)

    bvals_for_diffs = np.tile(bvals.reshape(bvals.shape[0], 1),
                              (1, centroids.shape[0]))

    shell_indices = np.argmin(np.abs(bvals_for_diffs - centroids), axis=1)

    return centroids, shell_indices


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input',
                   help='Path of the input multi-shell (multi-bvalue) diffusion dataset.')
    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')

    add_overwrite_arg(p)
    p.add_argument(
        '--mask',
        help='Path to a binary mask.\nOnly data inside the mask will be used '
             'for computations and reconstruction. (Default: None)')

    p.add_argument('--tolerance', '-t',
                   metavar='INT', type=int, default=20,
                   help='The tolerated distance between the b-values to '
                   'extract\nand the actual b-values [Default: %(default)s].')
    p.add_argument(
        '--min_k', dest='min_k', type=float, default='0',
        help='Minium kurtosis value in the output maps (ak, mk, rk). ' +
        '\nIn theory, -3/7 is the min kurtosis limit for regions that consist ' +
        '\nof water confined to spherical pores (see DIPY example and documentation) [Default: %(default)s].')
    p.add_argument(
        '--max_k', dest='max_k', type=float, default='3',
        help='Maximum kurtosis value in the output maps (ak, mk, rk). ' +
        '\nIn theory, 10 is the max kurtosis limit for regions that consist ' +
        '\nof water confined to spherical pores (see DIPY example and documentation) [Default: %(default)s].')
    p.add_argument(
        '--smooth', dest='smooth', type=float, default='1.25', 
        help='Smooth input DWI with a 3D Gaussian filter with ' +
        '\nfull-width-half-max (fwhm). Kurtosis fitting is sensitive and outliers occur easily. '+
        '\n This smoothing is thus done by default but can be turned off with fwhm=0. [Default: %(default)s].')
    p.add_argument(
        '--not_all', action='store_true', dest='not_all',
        help='If set, will only save the metrics explicitly specified using '
             'the other metrics flags. [Default: not set].')

    g = p.add_argument_group(title='Metrics files flags')
    g.add_argument('--ad', dest='ad', metavar='file', default='',
                   help='Output filename for the axial diffusivity from DKI.')
    g.add_argument('--ak', dest='ak', metavar='file', default='',
                   help='Output filename for the axial kurtosis.')
    g.add_argument(
        '--fa', dest='fa', metavar='file', default='',
        help='Output filename for the fractional anisotropy from DKI.')
    g.add_argument(
        '--md', dest='md', metavar='file', default='',
        help='Output filename for the mean diffusivity from DKI.')
    g.add_argument(
        '--mk', dest='mk', metavar='file', default='',
        help='Output filename for the mean kurtosis.')
    g.add_argument(
        '--rd', dest='rd', metavar='file', default='',
        help='Output filename for the radial diffusivity from DKI.')
    g.add_argument('--rk', dest='rk', metavar='file', default='',
                   help='Output filename for the radial kurtosis.')
    
    g = p.add_argument_group(title='Quality control files flags')
    g.add_argument(
        '--residual', dest='residual', metavar='file', default='',
        help='Output filename for the map of the residual of the tensor fit.')

    add_force_b0_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    if not args.not_all:
        args.fa = args.fa or 'dki_fa.nii.gz'
        args.md = args.md or 'dki_md.nii.gz'
        args.ad = args.ad or 'dki_ad.nii.gz'
        args.rd = args.rd or 'dki_rd.nii.gz'
        args.mk = args.mk or 'mk.nii.gz'
        args.rk = args.rk or 'rk.nii.gz'
        args.ak = args.ak or 'ak.nii.gz'
        args.residual = args.residual or 'dki_residual.nii.gz'
        
    outputs = [args.fa, args.md, args.ad, args.rd,
               args.mk, args.rk, args.ak, args.residual]

    if args.not_all and not any(outputs):
        parser.error('When using --not_all, you need to specify at least ' +
                     'one metric to output.')

    assert_inputs_exist(
        parser, [args.input, args.bvals, args.bvecs], [args.mask])
    assert_outputs_exists(parser, args, outputs)

    img = nib.load(args.input)
    data = img.get_data()
    affine = img.get_affine()
    if args.mask is None:
        mask = None
    else:
        mask = nib.load(args.mask).get_data().astype(np.bool)

    # Validate bvals and bvecs
    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    # Find the volume indices that correspond to the shells to extract.
    tol = args.tolerance
    shells, _ = identify_shells(bvals, tol)
    #print(shells, len(shells))
    if not len(shells) > 3 :
        parser.error('Data is not multi-shell. You need at least 2 non-zero b-values')
    
    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    check_b0_threshold(args, bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    
    fwhm = args.smooth
    if fwhm != 0 :
        gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
        data_smooth = np.zeros(data.shape)
        for v in range(data.shape[-1]):
            data_smooth[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)
        data = data_smooth

    # Compute DKI
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(data_smooth, mask=mask)

    FA = dkifit.fa
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    MD = dkifit.md
    AD = dkifit.ad
    RD = dkifit.rd

    min_k = args.min_k
    max_k = args.max_k

    MK = dkifit.mk(min_k, max_k)
    AK = dkifit.ak(min_k, max_k)
    RK = dkifit.rk(min_k, max_k)
        

    if args.fa:
        fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
        nib.save(fa_img, args.fa)

    if args.md:
        md_img = nib.Nifti1Image(MD.astype(np.float32), affine)
        nib.save(md_img, args.md)
        
    if args.ad:
        ad_img = nib.Nifti1Image(AD.astype(np.float32), affine)
        nib.save(ad_img, args.ad)

    if args.rd:
        rd_img = nib.Nifti1Image(RD.astype(np.float32), affine)
        nib.save(rd_img, args.rd)

    if args.mk:
        mk_img = nib.Nifti1Image(MK.astype(np.float32), affine)
        nib.save(mk_img, args.mk)

    if args.ak:
        ak_img = nib.Nifti1Image(AK.astype(np.float32), affine)
        nib.save(ak_img, args.ak)

    if args.rk:
        rk_img = nib.Nifti1Image(RK.astype(np.float32), affine)
        nib.save(rk_img, args.rk)


    if args.residual:
        S0 = np.mean(data[..., gtab.b0s_mask], axis=-1)
        data_p = dkifit.predict(gtab, S0)
        R = np.mean(np.abs(data_p[..., ~gtab.b0s_mask] -
                           data[..., ~gtab.b0s_mask]), axis=-1)

        norm = np.linalg.norm(R)
        if norm != 0: 
            R = R / norm
        
        if args.mask is not None:
            R *= mask
        
        R_img = nib.Nifti1Image(R.astype(np.float32), affine)
        nib.save(R_img, args.residual)


if __name__ == "__main__":
    main()
