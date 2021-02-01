#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute per-vertices hemisphere-aware (asymmetric) Gaussian
filtering of spherical functions (SF) given an array of spherical harmonics
(SH) coefficients. SF are filtered using a first-neighbor Gaussian filter.
Sphere directions are also weighted by their dot product with the direction
to the center of each neighbor, clipping to 0 negative values.

The argument `sigma` controls the standard deviation of the Gaussian. The
argument `sharpness` controls the exponent of the cosine weights. The higher it
is, the faster the weights of misaligned sphere directions decrease. A
sharpness of 0 gives the same weight to all sphere directions in an hemisphere.
Both `sharpness` and `sigma` must be positive.

The resulting SF can be expressed using a full SH basis or a symmetric SH basis
(where the effect of the filtering is a simple denoising). When a full SH basis
is used, an asymmetry map is also generated using an asymmetry measure (Cetin
Karayumak et al, 2018). The script also generates a mask of voxel higher than
`mask_eps` from the input image which can later be used to mask the output.

Using default parameters, the script completes in about 15-20 minutes for a
HCP subject fiber ODF processed with tractoflow. Also note the bigger the
sphere used for SH to SF projection, the higher the RAM consumption and
compute time.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from dipy.data import SPHERE_FILES
from dipy.reconst.shm import order_from_ncoef, sph_harm_ind_list

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             add_sh_basis_args,
                             assert_outputs_exist)

from scilpy.denoise.asym_enhancement import local_asym_gaussian_filtering


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_sh',
                   help='Path to the input file.')

    p.add_argument('out_sh',
                   help='File name for averaged signal.')

    p.add_argument('--out_asymmetry', default='asym_map.nii.gz',
                   help='File name for asymmetry map. Can only be outputed'
                        ' when the output SH basis is full. [%(default)s]')

    p.add_argument('--out_mask', default='mask.nii.gz',
                   help='File name for output mask. [%(default)s]')

    p.add_argument('--mask_eps', default=1e-16,
                   help='Threshold on SH coefficients norm for output mask.'
                        ' [%(default)s]')

    p.add_argument('--sh_order', default=8, type=int,
                   help='SH order of the input. [%(default)s]')

    add_sh_basis_args(p)

    p.add_argument('--sphere', default='repulsion724',
                   choices=sorted(SPHERE_FILES.keys()),
                   help='Sphere used for the SH to SF projection. '
                        '[%(default)s]')

    p.add_argument('--sharpness', default=1.0, type=float,
                   help='Specify sharpness factor to use for weighted average.'
                        ' [%(default)s]')

    p.add_argument('--sigma', default=1.0, type=float,
                   help='Sigma of the gaussian to use. [%(default)s]')

    p.add_argument('--out_sym', action='store_true',
                   help='If set, saves output in symmetric SH basis.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def generate_mask(sh, threshold):
    norm = np.linalg.norm(sh, axis=-1)
    mask = norm > threshold
    return mask


def compute_asymmetry_map(sh_coeffs):
    order = order_from_ncoef(sh_coeffs.shape[-1], full_basis=True)
    _, l_list = sph_harm_ind_list(order, full_basis=True)

    sign = np.power(-1.0, l_list)
    sign = np.reshape(sign, (1, 1, 1, len(l_list)))
    sh_squared = sh_coeffs**2
    mask = sh_squared.sum(axis=-1) > 0.

    asym_map = np.zeros(sh_coeffs.shape[:-1])
    asym_map[mask] = np.sum(sh_squared * sign, axis=-1)[mask] / \
        np.sum(sh_squared, axis=-1)[mask]

    asym_map = np.sqrt(1 - asym_map**2) * mask

    return asym_map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    outputs = [args.out_sh, args.out_mask]
    if not args.out_sym:
        outputs.append(args.out_asymmetry)

    # Checking args
    assert_outputs_exist(parser, args, outputs)
    assert_inputs_exist(parser, args.in_sh)

    # Prepare data
    sh_img = nib.load(args.in_sh)
    data = sh_img.get_fdata(dtype=np.float32)

    logging.info('Executing locally asymmetric Gaussian filtering.')
    filtered_sh = local_asym_gaussian_filtering(
        data, sh_order=args.sh_order,
        sh_basis=args.sh_basis,
        out_full_basis=not(args.out_sym),
        sphere_str=args.sphere,
        dot_sharpness=args.sharpness,
        sigma=args.sigma)

    logging.info('Saving filtered SH to file {0}.'.format(args.out_sh))
    nib.save(nib.Nifti1Image(filtered_sh, sh_img.affine), args.out_sh)

    # Save asymmetry measure map when the output is in full SH basis
    if args.out_sym:
        logging.info('Skipping asymmetry map because output is symmetric.')
    else:
        logging.info('Generating asymmetry map from output.')
        asym_map = compute_asymmetry_map(filtered_sh)
        logging.info('Saving asymmetry map to file '
                     '{0}.'.format(args.out_asymmetry))
        nib.save(nib.Nifti1Image(asym_map, sh_img.affine),
                 args.out_asymmetry)

    # Generate mask by applying threshold on input SH
    logging.info('Generating mask by thresholding input SH.')
    mask = generate_mask(data, args.mask_eps)
    logging.info('Saving mask to file {0}.'.format(args.out_mask))
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), sh_img.affine),
             args.out_mask)


if __name__ == "__main__":
    main()
