#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute the Diffusion Kurtosis Imaging (DKI) and Mean Signal DKI
(MSDKI) metrics. DKI is a multi-shell diffusion model. The input DWI needs
to be multi-shell, i.e. multi-bvalued.

Since the diffusion kurtosis model involves the estimation of a large number
of parameters and since the non-Gaussian components of the diffusion signal
are more sensitive to artefacts, you should really denoise your DWI volume
before using this DKI script (e.g. scil_denoising_nlmeans.py). Moreover, to
remove biases due to fiber dispersion, fiber crossings and other mesoscopic
properties of the underlying tissue, MSDKI does a powder-average of DWI for all
directions, thus removing the orientational dependencies and creating an
alternative mean kurtosis map.

DKI is also known to be vulnerable to artefacted voxels induced by the
low radial diffusivities of aligned white matter (CC, CST voxels). Since it is
very hard to capture non-Gaussian information due to the low decays in radial
direction, its kurtosis estimates have very low robustness.
Noisy kurtosis estimates tend to be negative and its absolute values can have
order of magnitudes higher than the typical kurtosis values. Consequently,
these negative kurtosis values will heavily propagate to the mean and radial
kurtosis metrics. This is well-reported in [Rafael Henriques MSc thesis 2012,
chapter 3]. Two ways to overcome this issue: i) compute the kurtosis values
from powder-averaged MSDKI, and ii) perform 3D Gaussian smoothing. On
powder-averaged signal decays, you don't have this low diffusivity issue and
your kurtosis estimates have much higher precision (additionally they are
independent to the fODF).

By default, will output all available metrics, using default names. Specific
names can be specified using the metrics flags that are listed in the "Metrics
files flags" section. If --not_all is set, only the metrics specified
explicitly by the flags will be output.

This script directly comes from the DIPY example gallery and references
therein.
[1] examples_built/reconst_dki/#example-reconst-dki
[2] examples_built/reconst_msdki/#example-reconst-msdki

Formerly: scil_compute_kurtosis_metrics.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

import dipy.reconst.dki as dki
import dipy.reconst.msdki as msdki

from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

from scipy.ndimage import gaussian_filter

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_force_b0_arg,
                             add_verbose_arg)
from scilpy.gradients.bvec_bval_tools import (normalize_bvecs,
                                              is_normalized_bvecs,
                                              check_b0_threshold,
                                              identify_shells)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='Path of the input multi-shell DWI dataset.')
    p.add_argument('in_bval',
                   help='Path of the b-value file, in FSL format.')
    p.add_argument('in_bvec',
                   help='Path of the b-vector file, in FSL format.')

    p.add_argument('--mask',
                   help='Path to a binary mask.' +
                   '\nOnly data inside the mask will be used '
                   'for computations and reconstruction. ' +
                   '\n[Default: None]')

    p.add_argument('--tolerance', '-t',
                   metavar='INT', type=int, default=20,
                   help='The tolerated distance between the b-values to '
                   'extract\nand the actual b-values [Default: %(default)s].')
    p.add_argument('--min_k', type=float, default=0.0,
                   help='Minimum kurtosis value in the output maps ' +
                   '\n(ak, mk, rk). In theory, -3/7 is the min kurtosis ' +
                   '\nlimit for regions that consist of water confined ' +
                   '\nto spherical pores (see DIPY example and ' +
                   '\ndocumentation) [Default: %(default)s].')
    p.add_argument('--max_k', type=float, default=3.0,
                   help='Maximum kurtosis value in the output maps ' +
                   '\n(ak, mk, rk). In theory, 10 is the max kurtosis' +
                   '\nlimit for regions that consist of water confined' +
                   '\nto spherical pores (see DIPY example and ' +
                   '\ndocumentation) [Default: %(default)s].')
    p.add_argument('--smooth', type=float, default=2.5,
                   help='Smooth input DWI with a 3D Gaussian filter with ' +
                   '\nfull-width-half-max (fwhm). Kurtosis fitting is ' +
                   '\nsensitive and outliers occur easily. According to' +
                   '\ntests on HCP, CB_Brain, Penthera3T, this smoothing' +
                   '\nis thus turned ON by default with fwhm=2.5. ' +
                   '\n[Default: %(default)s].')
    p.add_argument('--not_all', action='store_true',
                   help='If set, will only save the metrics explicitly ' +
                   '\nspecified using the other metrics flags. ' +
                   '\n[Default: not set].')

    g = p.add_argument_group(title='Metrics files flags')
    g.add_argument('--ak', metavar='file', default='',
                   help='Output filename for the axial kurtosis.')
    g.add_argument('--mk', metavar='file', default='',
                   help='Output filename for the mean kurtosis.')
    g.add_argument('--rk', metavar='file', default='',
                   help='Output filename for the radial kurtosis.')
    g.add_argument('--msk', metavar='file', default='',
                   help='Output filename for the mean signal kurtosis.')
    g.add_argument('--dki_fa', metavar='file', default='',
                   help='Output filename for the fractional anisotropy ' +
                   'from DKI.')
    g.add_argument('--dki_md', metavar='file', default='',
                   help='Output filename for the mean diffusivity from DKI.')
    g.add_argument('--dki_ad', metavar='file', default='',
                   help='Output filename for the axial diffusivity from DKI.')
    g.add_argument('--dki_rd', metavar='file', default='',
                   help='Output filename for the radial diffusivity from DKI.')

    g = p.add_argument_group(title='Quality control files flags')
    g.add_argument('--dki_residual', metavar='file', default='',
                   help='Output filename for the map of the residual ' +
                   'of the tensor fit.')
    g.add_argument('--msd', metavar='file', default='',
                   help='Output filename for the mean signal diffusion ' +
                   '(powder-average).')

    add_verbose_arg(p)
    add_force_b0_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    logger = logging.getLogger("Compute_DKI_Metrics")
    logger.setLevel(logging.INFO)

    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.not_all:
        args.dki_fa = args.dki_fa or 'dki_fa.nii.gz'
        args.dki_md = args.dki_md or 'dki_md.nii.gz'
        args.dki_ad = args.dki_ad or 'dki_ad.nii.gz'
        args.dki_rd = args.dki_rd or 'dki_rd.nii.gz'
        args.mk = args.mk or 'mk.nii.gz'
        args.rk = args.rk or 'rk.nii.gz'
        args.ak = args.ak or 'ak.nii.gz'
        args.dki_residual = args.dki_residual or 'dki_residual.nii.gz'
        args.msk = args.msk or 'msk.nii.gz'
        args.msd = args.msd or 'msd.nii.gz'

    outputs = [args.dki_fa, args.dki_md, args.dki_ad, args.dki_rd,
               args.mk, args.rk, args.ak, args.dki_residual,
               args.msk, args.msd]

    if args.not_all and not any(outputs):
        parser.error('When using --not_all, you need to specify at least ' +
                     'one metric to output.')

    assert_inputs_exist(
        parser, [args.in_dwi, args.in_bval, args.in_bvec], args.mask)
    assert_outputs_exist(parser, args, outputs)

    img = nib.load(args.in_dwi)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    if args.mask is None:
        mask = None
    else:
        mask_img = nib.load(args.mask)
        mask = get_data_as_mask(mask_img, dtype=bool)

    # Validate bvals and bvecs
    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    # Find the volume indices that correspond to the shells to extract.
    tol = args.tolerance
    shells, _ = identify_shells(bvals, tol)
    if not len(shells) >= 3:
        parser.error('Data is not multi-shell. You need at least 2 non-zero' +
                     ' b-values')

    if (shells > 2500).any():
        logging.warning('You seem to be using b > 2500 s/mm2 DWI data. ' +
                        'In theory, this is beyond the optimal range for DKI')

    b0_thr = check_b0_threshold(args, bvals.min(), bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)

    fwhm = args.smooth
    if fwhm > 0:
        # converting fwhm to Gaussian std
        gauss_std = fwhm / np.sqrt(8 * np.log(2))
        data_smooth = np.zeros(data.shape)
        for v in range(data.shape[-1]):
            data_smooth[..., v] = gaussian_filter(data[..., v],
                                                  sigma=gauss_std)
        data = data_smooth

    # Compute DKI
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(data, mask=mask)

    min_k = args.min_k
    max_k = args.max_k

    if args.dki_fa:
        FA = dkifit.fa
        FA[np.isnan(FA)] = 0
        FA = np.clip(FA, 0, 1)

        fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
        nib.save(fa_img, args.dki_fa)

    if args.dki_md:
        MD = dkifit.md
        md_img = nib.Nifti1Image(MD.astype(np.float32), affine)
        nib.save(md_img, args.dki_md)

    if args.dki_ad:
        AD = dkifit.ad
        ad_img = nib.Nifti1Image(AD.astype(np.float32), affine)
        nib.save(ad_img, args.dki_ad)

    if args.dki_rd:
        RD = dkifit.rd
        rd_img = nib.Nifti1Image(RD.astype(np.float32), affine)
        nib.save(rd_img, args.dki_rd)

    if args.mk:
        MK = dkifit.mk(min_k, max_k)
        mk_img = nib.Nifti1Image(MK.astype(np.float32), affine)
        nib.save(mk_img, args.mk)

    if args.ak:
        AK = dkifit.ak(min_k, max_k)
        ak_img = nib.Nifti1Image(AK.astype(np.float32), affine)
        nib.save(ak_img, args.ak)

    if args.rk:
        RK = dkifit.rk(min_k, max_k)
        rk_img = nib.Nifti1Image(RK.astype(np.float32), affine)
        nib.save(rk_img, args.rk)

    if args.msk or args.msd:
        # Compute MSDKI
        msdki_model = msdki.MeanDiffusionKurtosisModel(gtab)
        msdki_fit = msdki_model.fit(data, mask=mask)

        if args.msk:
            MSK = msdki_fit.msk
            MSK[np.isnan(MSK)] = 0
            MSK = np.clip(MSK, min_k, max_k)

            msk_img = nib.Nifti1Image(MSK.astype(np.float32), affine)
            nib.save(msk_img, args.msk)

        if args.msd:
            MSD = msdki_fit.msd
            msd_img = nib.Nifti1Image(MSD.astype(np.float32), affine)
            nib.save(msd_img, args.msd)

    if args.dki_residual:
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
        nib.save(R_img, args.dki_residual)


if __name__ == "__main__":
    main()
