#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to denoise a dataset with the Non-Local Means algorithm (from Dipy's
package).

This method requires an estimate of the noise (sigma). There are three methods
available:
    1. Provide your own sigma value, if you already know it.
    2. Estimate it through Dipy's "noise_estimate" method.
    Here are Dipy's note on this method:
     > This function is the same as manually taking the standard deviation of
     > the background and gives one value for each 3D volume. It also includes
     > the coil-dependent correction factor of Koay 2006 (see [1], equation 18)
     > with theta = 0. Since this function was introduced in [2] for T1
     > imaging, it is expected to perform ok on diffusion MRI data, but might
     > oversmooth some regions and leave others un-denoised for spatially
     > varying noise profiles. Consider using --piesno to estimate sigma
     > instead if visual inaccuracies are apparent in the denoised result.
    (Note. We then use the median of the noise for all volumes)
    3. Estimate it through using Dipy's "Piesno" method, as described in [3].
    Here are Dipy's note on this method:
     > It is expected that
     >   1. The data has a noisy, non-masked background and
     >   2. The data is a repetition of the same measurements along the last
     >      axis, i.e. dMRI or fMRI data, not structural data like T1 or T2
     >      images.

The two last methods require the number of coils in the data to be known.
Typically, if you know that the noise follows a Gaussian distribution, you
may use --number_coils=0.

References:
[1] Koay, C. G., & Basser, P. J. (2006). Analytically exact correction scheme
for signal extraction from noisy magnitude MR signals. Journal of Magnetic
Resonance, 179(2), 317-22.
[2] Coupe, P., Yger, P., Prima, S., Hellier, P., Kervrann, C., Barillot,
C., (2008). An optimized blockwise nonlocal means denoising filter for 3-D
magnetic resonance images, IEEE Trans. Med. Imaging 27, 425-41.
[3] St-Jean, S., Coupé, P., & Descoteaux, M. (2016). Non Local Spatial and
Angular Matching: Enabling higher spatial resolution diffusion MRI datasets
through adaptive denoising. Medical image analysis, 32, 115-130.
[4] Koay CG, Ozarslan E and Pierpaoli C. "Probabilistic Identification and
Estimation of Noise (PIESNO): A self-consistent approach and its applications
in MRI." Journal of Magnetic Resonance 2009; 199: 94-103.

Formerly: scil_run_nlmeans.py

"""
import argparse
import logging
import warnings

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
import nibabel as nib
import numpy as np

from scilpy.image.volume_metrics import estimate_piesno_sigma
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_processes_arg,
                             add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_headers_compatible, ranged_type)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_image',
                   help='Path of the image file to denoise (3D or 4D data)')
    p.add_argument('out_image',
                   help='Path to save the denoised image file.')
    p.add_argument('--mask_denoise',
                   help="Path to a binary mask. Only the data inside the mask "
                        "will be denoised. If not provided, only non-zero "
                        "voxels will be denoised.")
    p.add_argument('--gaussian', action='store_true',
                   help="If you know that your data contains gaussian noise, "
                        "use this option. Otherwise, Rician is assumed.")

    g = p.add_argument_group("Noise estimation methods")
    g = g.add_mutually_exclusive_group(required=True)
    g.add_argument(
        '--sigma', metavar='float', type=float,
        help='Provide your own standard deviation of the noise.')
    g.add_argument(
        '--basic_sigma', action='store_true',
        help="Use dipy's basic estimation of sigma.")
    g.add_argument(
        '--piesno', action='store_true',
        help="Estimate sigma using Piesno's method. If data is 4D, the noise "
             "is estimated for each slice (3rd dimension).")

    g = p.add_argument_group("Noise estimation options: piesno and basic")
    g.add_argument(
        '--number_coils', type=ranged_type(int, min_value=0),
        help="Option for Dipy's noise estimation. Here is their description:\n"
             ">Number of coils of the receiver array. Use N = 1 in case of a\n"
             ">SENSE reconstruction (Philips scanners) or the number of \n"
             ">coils for a GRAPPA reconstruction (Siemens and GE). Use 0 to \n"
             ">disable the correction factor, as for example if the noise is\n"
             ">Gaussian distributed. See [1] for more information.\n"
             "Note. If you don't know the number of coils, 0 will probably "
             "work.")

    g = p.add_argument_group("Noise estimation options: basic")
    gg = g.add_mutually_exclusive_group()
    gg.add_argument(
        '--mask_sigma',
        help='Path to a binary mask for --basic_sigma estimation. Only the '
             'data inside the mask will be used to estimate sigma. If not '
             'provided, only non-zero voxels will be used.')
    gg.add_argument(
        '--sigma_from_all_voxels', action='store_true',
        help="If set, all voxels are used for the --basic_sigma estimation, "
             "even zeros.")

    g = p.add_argument_group("Noise estimation options: piesno")
    g.add_argument('--save_piesno_mask', metavar='filepath',
                   help="If set, save piesno mask.")

    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    if (args.number_coils is not None and args.number_coils == 0 and
            not args.gaussian):
        logging.warning("Usually, with --number_coils 0, the data ìs "
                        "considered to have Gaussian noise, but you "
                        "did not select --gaussian. Proceed with care.")

    if (args.basic_sigma or args.piesno) and args.number_coils is None:
        parser.error("Please provide the number of coils for basic_sigma "
                     "and piesno options.")

    if args.piesno or args.sigma:
        if args.sigma_from_all_voxels:
            parser.error("You selected --sigma_from_all_voxels, but this is "
                         "only available for the --basic_sigma method.")
        if args.mask_sigma:
            parser.error("You selected --mask_sigma, but this is "
                         "only available for the --basic_sigma method.")

    if args.save_piesno_mask and not args.piesno:
        parser.error("Option --save_piesno_mask cannot be used when --pieno "
                     "is not selected.")

    assert_inputs_exist(parser, args.in_image,
                        [args.mask_denoise, args.mask_sigma])
    assert_outputs_exist(parser, args, args.out_image, args.save_piesno_mask)
    assert_headers_compatible(parser, args.in_image,
                              [args.mask_denoise, args.mask_sigma])

    # Loading
    vol = nib.load(args.in_image)
    vol_data = vol.get_fdata(dtype=np.float32)
    nb_volumes = 1 if (len(vol_data.shape) != 4 or vol_data.shape[3] == 1) \
        else vol_data.shape[-1]

    if args.piesno and nb_volumes == 1:
        parser.error("The piesno method requires 4D data.")

    # Denoising mask
    if args.mask_denoise is None:
        mask_denoise = np.zeros(vol_data.shape[0:3], dtype=bool)
        if vol_data.ndim == 4:
            mask_denoise[np.sum(vol_data, axis=-1) > 0] = True
        else:
            mask_denoise[vol_data > 0] = True
    else:
        mask_denoise = get_data_as_mask(nib.load(args.mask_denoise),
                                        dtype=bool)

    # Processing
    if args.sigma is not None:
        logging.info('User supplied noise standard deviation is {}'
                     .format(args.sigma))
        sigma = np.ones(vol_data.shape[:3]) * args.sigma
    elif args.basic_sigma:
        if args.mask_sigma:
            mask_sigma = get_data_as_mask(nib.load(args.mask_sigma))
            tmp_vol_data = (vol_data * mask_sigma[:, :, :, None]
                            ).astype(np.float32)
        else:
            tmp_vol_data = vol_data.astype(np.float32)

        logging.info('Estimating noise')
        sigma = estimate_sigma(
            tmp_vol_data,
            disable_background_masking=args.sigma_from_all_voxels,
            N=args.number_coils)
        logging.info(
            "The estimated noise for each volume is: {}".format(sigma))
        sigma = np.median(sigma)  # Managing 4D data.
        logging.info('The median noise is: {}'.format(sigma))

        # Broadcast the single value to a whole 3D volume for nlmeans
        sigma = np.ones(vol_data.shape) * sigma
    else:  # --piesno
        logging.info("Computing sigma: one value per slice.")
        sigma, mask_noise = estimate_piesno_sigma(vol_data, args.number_coils)

        if args.save_piesno_mask:
            logging.info("Saving resulting Piesno noise mask in {}"
                         .format(args.save_piesno_mask))
            nib.save(nib.Nifti1Image(mask_noise, vol.affine,
                                     header=vol.header),
                     args.save_piesno_mask)

        # Broadcast the values per slice to a whole 3D volume for nlmeans
        sigma = np.ones(vol_data.shape[:3]) * sigma[None, None, :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        data_denoised = nlmeans(
            vol_data, sigma, mask=mask_denoise, rician=not args.gaussian,
            num_threads=args.nbr_processes)

    # Saving
    nib.save(nib.Nifti1Image(data_denoised, vol.affine, header=vol.header),
             args.out_image)


if __name__ == "__main__":
    main()
