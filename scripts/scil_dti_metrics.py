#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute all of the Diffusion Tensor Imaging (DTI) metrics.

By default, will output all available metrics, using default names. Specific
names can be specified using the metrics flags that are listed in the "Metrics
files flags" section.

If --not_all is set, only the metrics specified explicitly by the flags
will be output. The available metrics are:

fractional anisotropy (FA), geodesic anisotropy (GA), axial diffusivisty (AD),
radial diffusivity (RD), mean diffusivity (MD), mode, red-green-blue colored
FA (rgb), principal tensor e-vector and tensor coefficients (dxx, dxy, dxz,
dyy, dyz, dzz).

For all the quality control metrics such as residual, physically implausible
signals, pulsation and misalignment artifacts, see
[J-D Tournier, S. Mori, A. Leemans. Diffusion Tensor Imaging and Beyond.
MRM 2011].
"""

import argparse
import logging

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
import dipy.denoise.noise_estimate as ne
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import (TensorModel, color_fa, fractional_anisotropy,
                              geodesic_anisotropy, mean_diffusivity,
                              axial_diffusivity, norm,
                              radial_diffusivity, lower_triangular)
# Aliased to avoid clashes with images called mode.
from dipy.reconst.dti import mode as dipy_mode

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg,
                             add_force_b0_arg)
from scilpy.io.tensor import convert_tensor_from_dipy_format, \
    supported_tensor_formats, tensor_format_description
from scilpy.gradients.bvec_bval_tools import (normalize_bvecs,
                                              is_normalized_bvecs,
                                              check_b0_threshold)
from scilpy.utils.filenames import add_filename_suffix, split_name_with_nii

logger = logging.getLogger("DTI_Metrics")
logger.setLevel(logging.INFO)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='Path of the input diffusion volume.')
    p.add_argument('in_bval',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('in_bvec',
                   help='Path of the bvecs file, in FSL format.')

    add_overwrite_arg(p)
    p.add_argument(
        '--mask',
        help='Path to a binary mask.\nOnly data inside the mask will be used '
             'for computations and reconstruction. (Default: %(default)s)')
    p.add_argument(
        '--method', dest='method', metavar='method_name', default='WLS',
        choices=['WLS', 'LS', 'NLLS', 'restore'],
        help='Tensor fit method.\nWLS for weighted least squares' +
             '\nLS for ordinary least squares' +
             '\nNLLS for non-linear least-squares' +
             '\nrestore for RESTORE robust tensor fitting. '
             '(Default: %(default)s)')
    p.add_argument(
        '--not_all', action='store_true', dest='not_all',
        help='If set, will only save the metrics explicitly specified using '
             'the other metrics flags. (Default: not set).')

    g = p.add_argument_group(title='Metrics files flags')
    g.add_argument('--ad', dest='ad', metavar='file', default='',
                   help='Output filename for the axial diffusivity.')
    g.add_argument(
        '--evecs', dest='evecs', metavar='file', default='',
        help='Output filename for the eigenvectors of the tensor.')
    g.add_argument(
        '--evals', dest='evals', metavar='file', default='',
        help='Output filename for the eigenvalues of the tensor.')
    g.add_argument(
        '--fa', dest='fa', metavar='file', default='',
        help='Output filename for the fractional anisotropy.')
    g.add_argument(
        '--ga', dest='ga', metavar='file', default='',
        help='Output filename for the geodesic anisotropy.')
    g.add_argument(
        '--md', dest='md', metavar='file', default='',
        help='Output filename for the mean diffusivity.')
    g.add_argument(
        '--mode', dest='mode', metavar='file', default='',
        help='Output filename for the mode.')
    g.add_argument(
        '--norm', dest='norm', metavar='file', default='',
        help='Output filename for the tensor norm.')
    g.add_argument(
        '--rgb', dest='rgb', metavar='file', default='',
        help='Output filename for the colored fractional anisotropy.')
    g.add_argument(
        '--rd', dest='rd', metavar='file', default='',
        help='Output filename for the radial diffusivity.')
    g.add_argument(
        '--tensor', dest='tensor', metavar='file', default='',
        help='Output filename for the tensor coefficients.')
    g.add_argument('--tensor_format', choices=supported_tensor_formats,
                   default='fsl',
                   help=("Format used for the tensors saved in --tensor file."
                         "(default: %(default)s)\n"
                         + tensor_format_description))

    g = p.add_argument_group(title='Quality control files flags')
    g.add_argument(
        '--non-physical', dest='p_i_signal', metavar='file', default='',
        help='Output filename for the voxels with physically implausible '
             'signals \nwhere the mean of b=0 images is below one or more '
             'diffusion-weighted images.')
    g.add_argument(
        '--pulsation', dest='pulsation', metavar='string', default='',
        help='Standard deviation map across all diffusion-weighted images '
             'and across b=0 images if more than one is available.\nShows '
             'pulsation and misalignment artifacts.')
    g.add_argument(
        '--residual', dest='residual', metavar='file', default='',
        help='Output filename for the map of the residual of the tensor fit.')

    add_verbose_arg(p)
    add_force_b0_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.not_all:
        args.fa = args.fa or 'fa.nii.gz'
        args.ga = args.ga or 'ga.nii.gz'
        args.rgb = args.rgb or 'rgb.nii.gz'
        args.md = args.md or 'md.nii.gz'
        args.ad = args.ad or 'ad.nii.gz'
        args.rd = args.rd or 'rd.nii.gz'
        args.mode = args.mode or 'mode.nii.gz'
        args.norm = args.norm or 'tensor_norm.nii.gz'
        args.tensor = args.tensor or 'tensor.nii.gz'
        args.evecs = args.evecs or 'tensor_evecs.nii.gz'
        args.evals = args.evals or 'tensor_evals.nii.gz'
        args.residual = args.residual or 'dti_residual.nii.gz'
        args.p_i_signal =\
            args.p_i_signal or 'physically_implausible_signals_mask.nii.gz'
        args.pulsation = args.pulsation or 'pulsation_and_misalignment.nii.gz'

    outputs = [args.fa, args.ga, args.rgb, args.md, args.ad, args.rd,
               args.mode, args.norm, args.tensor, args.evecs, args.evals,
               args.residual, args.p_i_signal, args.pulsation]
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
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)

    # Validate bvals and bvecs
    logging.info('Tensor estimation with the {} method...'.format(args.method))
    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    b0_thr = check_b0_threshold(
        args.force_b0_threshold, bvals.min(), bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)

    # Get tensors
    if args.method == 'restore':
        sigma = ne.estimate_sigma(data)
        tenmodel = TensorModel(gtab, fit_method=args.method, sigma=sigma,
                               min_signal=np.min(data[data > 0]))
    else:
        tenmodel = TensorModel(gtab, fit_method=args.method,
                               min_signal=np.min(data[data > 0]))

    tenfit = tenmodel.fit(data, mask)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    if args.tensor:
        # Get the Tensor values
        # Format them for visualization in various software.
        tensor_vals = lower_triangular(tenfit.quadratic_form)
        tensor_vals_reordered = convert_tensor_from_dipy_format(
            tensor_vals, final_format=args.tensor_format)

        fiber_tensors = nib.Nifti1Image(
            tensor_vals_reordered.astype(np.float32), affine)
        nib.save(fiber_tensors, args.tensor)

        del tensor_vals, fiber_tensors, tensor_vals_reordered

    if args.fa:
        fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
        nib.save(fa_img, args.fa)

        del fa_img

    if args.ga:
        GA = geodesic_anisotropy(tenfit.evals)
        GA[np.isnan(GA)] = 0

        ga_img = nib.Nifti1Image(GA.astype(np.float32), affine)
        nib.save(ga_img, args.ga)

        del GA, ga_img

    if args.rgb:
        RGB = color_fa(FA, tenfit.evecs)
        rgb_img = nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine)
        nib.save(rgb_img, args.rgb)

        del RGB, rgb_img

    if args.md:
        MD = mean_diffusivity(tenfit.evals)
        md_img = nib.Nifti1Image(MD.astype(np.float32), affine)
        nib.save(md_img, args.md)

        del MD, md_img

    if args.ad:
        AD = axial_diffusivity(tenfit.evals)
        ad_img = nib.Nifti1Image(AD.astype(np.float32), affine)
        nib.save(ad_img, args.ad)

        del AD, ad_img

    if args.rd:
        RD = radial_diffusivity(tenfit.evals)
        rd_img = nib.Nifti1Image(RD.astype(np.float32), affine)
        nib.save(rd_img, args.rd)

        del RD, rd_img

    if args.mode:
        # Compute tensor mode
        inter_mode = dipy_mode(tenfit.quadratic_form)

        # Since the mode computation can generate NANs when not masked,
        # we need to remove them.
        non_nan_indices = np.isfinite(inter_mode)
        mode = np.zeros(inter_mode.shape)
        mode[non_nan_indices] = inter_mode[non_nan_indices]

        mode_img = nib.Nifti1Image(mode.astype(np.float32), affine)
        nib.save(mode_img, args.mode)

        del inter_mode, mode_img, mode

    if args.norm:
        NORM = norm(tenfit.quadratic_form)
        norm_img = nib.Nifti1Image(NORM.astype(np.float32), affine)
        nib.save(norm_img, args.norm)

        del NORM, norm_img

    if args.evecs:
        evecs = tenfit.evecs.astype(np.float32)
        evecs_img = nib.Nifti1Image(evecs, affine)
        nib.save(evecs_img, args.evecs)

        # save individual e-vectors also
        for i in range(3):
            e_img = nib.Nifti1Image(evecs[..., i], affine)
            nib.save(e_img, add_filename_suffix(args.evecs, '_v'+str(i+1)))
            del e_img

        del evecs, evecs_img

    if args.evals:
        evals = tenfit.evals.astype(np.float32)
        evals_img = nib.Nifti1Image(evals, affine)
        nib.save(evals_img, args.evals)

        # save individual e-values also
        for i in range(3):
            e_img = nib.Nifti1Image(evals[..., i], affine)
            nib.save(e_img, add_filename_suffix(args.evals, '_e' + str(i+1)))
            del e_img

        del evals, evals_img

    if args.p_i_signal:
        S0 = np.mean(data[..., gtab.b0s_mask], axis=-1, keepdims=True)
        DWI = data[..., ~gtab.b0s_mask]
        pis_mask = np.max(S0 < DWI, axis=-1)

        if args.mask is not None:
            pis_mask *= mask

        pis_img = nib.Nifti1Image(pis_mask.astype(np.int16), affine)
        nib.save(pis_img, args.p_i_signal)

        del pis_img, S0, DWI

    if args.pulsation:
        STD = np.std(data[..., ~gtab.b0s_mask], axis=-1)

        if args.mask is not None:
            STD *= mask

        std_img = nib.Nifti1Image(STD.astype(np.float32), affine)
        nib.save(std_img, add_filename_suffix(args.pulsation, '_std_dwi'))

        if np.sum(gtab.b0s_mask) <= 1:
            logger.info('Not enough b=0 images to output standard '
                        'deviation map')
        else:
            if len(np.where(gtab.b0s_mask)) == 2:
                logger.info('Only two b=0 images. Be careful with the '
                            'interpretation of this std map')

            STD = np.std(data[..., gtab.b0s_mask], axis=-1)

            if args.mask is not None:
                STD *= mask

            std_img = nib.Nifti1Image(STD.astype(np.float32), affine)
            nib.save(std_img, add_filename_suffix(args.pulsation, '_std_b0'))

        del STD, std_img

    if args.residual:
        # Mean residual image
        S0 = np.mean(data[..., gtab.b0s_mask], axis=-1)
        data_diff = np.zeros(data.shape, dtype=np.float32)

        for i in range(data.shape[0]):
            if args.mask is not None:
                tenfit2 = tenmodel.fit(data[i, :, :, :], mask[i, :, :])
            else:
                tenfit2 = tenmodel.fit(data[i, :, :, :])

            data_diff[i, :, :, :] = np.abs(
                tenfit2.predict(gtab, S0[i, :, :]).astype(
                    np.float32) - data[i, :, :])

        R = np.mean(data_diff[..., ~gtab.b0s_mask], axis=-1, dtype=np.float32)

        if args.mask is not None:
            R *= mask

        R_img = nib.Nifti1Image(R.astype(np.float32), affine)
        nib.save(R_img, args.residual)

        del R, R_img, S0

        # Each volume's residual statistics
        if args.mask is None:
            logger.info("Outlier detection will not be performed, since no "
                        "mask was provided.")

        stats = [dict.fromkeys(['label', 'mean', 'iqr', 'cilo', 'cihi',
                                'whishi', 'whislo', 'fliers', 'q1',
                                'med', 'q3'], [])
                 for i in range(data.shape[-1])]
        # Note that stats will be computed manually and plotted using bxp
        # but could be computed using stats = cbook.boxplot_stats
        # or pyplot.boxplot(x)
        # mean residual per DWI
        R_k = np.zeros(data.shape[-1], dtype=np.float32)
        # std residual per DWI
        std = np.zeros(data.shape[-1], dtype=np.float32)
        # first quartile per DWI
        q1 = np.zeros(data.shape[-1], dtype=np.float32)
        # third quartile per DWI
        q3 = np.zeros(data.shape[-1], dtype=np.float32)
        # interquartile per DWI
        iqr = np.zeros(data.shape[-1], dtype=np.float32)
        percent_outliers = np.zeros(data.shape[-1], dtype=np.float32)
        nb_voxels = np.count_nonzero(mask)
        for k in range(data.shape[-1]):
            x = data_diff[..., k]
            R_k[k] = np.mean(x)
            std[k] = np.std(x)
            q3[k], q1[k] = np.percentile(x, [75, 25])
            iqr[k] = q3[k] - q1[k]
            stats[k]['med'] = (q1[k] + q3[k]) / 2
            stats[k]['mean'] = R_k[k]
            stats[k]['q1'] = q1[k]
            stats[k]['q3'] = q3[k]
            stats[k]['whislo'] = q1[k] - 1.5 * iqr[k]
            stats[k]['whishi'] = q3[k] + 1.5 * iqr[k]
            stats[k]['label'] = k

            # Outliers are observations that fall below Q1 - 1.5(IQR) or
            # above Q3 + 1.5(IQR) We check if a voxel is an outlier only if
            # we have a mask, else we are biased.
            if args.mask is not None:
                outliers = (x < stats[k]['whislo']) | (x > stats[k]['whishi'])
                percent_outliers[k] = np.sum(outliers)/nb_voxels*100
                # What would be our definition of too many outliers?
                # Maybe mean(all_means)+-3SD?
                # Or we let people choose based on the figure.
                # if percent_outliers[k] > ???? :
                #    logger.warning('   Careful! Diffusion-Weighted Image'
                #                   ' i=%s has %s %% outlier voxels',
                #                   k, percent_outliers[k])

        # Saving all statistics as npy values
        residual_basename, _ = split_name_with_nii(args.residual)
        res_stats_basename = residual_basename + ".npy"
        np.save(add_filename_suffix(
            res_stats_basename, "_mean_residuals"), R_k)
        np.save(add_filename_suffix(res_stats_basename, "_q1_residuals"), q1)
        np.save(add_filename_suffix(res_stats_basename, "_q3_residuals"), q3)
        np.save(add_filename_suffix(res_stats_basename, "_iqr_residuals"), iqr)
        np.save(add_filename_suffix(res_stats_basename, "_std_residuals"), std)

        # Showing results in graph
        if args.mask is None:
            fig, axe = plt.subplots(nrows=1, ncols=1, squeeze=False)
        else:
            fig, axe = plt.subplots(nrows=1, ncols=2, squeeze=False,
                                    figsize=[10, 4.8])
            # Default is [6.4, 4.8]. Increasing width to see better.

        medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
        meanprops = dict(linestyle='-', linewidth=2.5, color='green')
        axe[0, 0].bxp(stats, showmeans=True, meanline=True, showfliers=False,
                      medianprops=medianprops, meanprops=meanprops)
        axe[0, 0].set_xlabel('DW image')
        axe[0, 0].set_ylabel('Residuals per DWI volume. Red is median,\n'
                             'green is mean. Whiskers are 1.5*interquartile')
        axe[0, 0].set_title('Residuals')
        axe[0, 0].set_xticks(range(0, q1.shape[0], 5))
        axe[0, 0].set_xticklabels(range(0, q1.shape[0], 5))

        if args.mask is not None:
            axe[0, 1].plot(range(data.shape[-1]), percent_outliers)
            axe[0, 1].set_xticks(range(0, q1.shape[0], 5))
            axe[0, 1].set_xticklabels(range(0, q1.shape[0], 5))
            axe[0, 1].set_xlabel('DW image')
            axe[0, 1].set_ylabel('Percentage of outlier voxels')
            axe[0, 1].set_title('Outliers')
        plt.savefig(residual_basename + '_residuals_stats.png')


if __name__ == "__main__":
    main()
