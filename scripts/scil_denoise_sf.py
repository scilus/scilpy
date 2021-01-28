#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute per-vertices hemisphere-aware (asymmetric) Gaussian
filtering of spherical functions (SF) given an array of spherical harmonics
(SH) coefficients.

The resulting SF can be expressed using a full SH basis (to keep the
asymmetry resulting from the filtering) or a symmetric SH basis (where the
effect of the filtering is a simple denoising).
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np

from dipy.data import SPHERE_FILES

from scilpy.io.utils import (add_overwrite_arg,
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

    p.add_argument('--out_mask', default='mask.nii.gz',
                   help='File name for output mask. [%(default)s]')

    p.add_argument('--mask_eps', default=1e-16,
                   help='Threshold on SH coefficients norm for output mask.'
                   ' [%(default)s]')

    p.add_argument('--sh_order', default=8, type=int,
                   help='SH order of the input. [%(default)s]')

    p.add_argument('--sphere', default='repulsion724',
                   choices=sorted(SPHERE_FILES.keys()),
                   help='Sphere used for the SH projection. [%(default)s]')

    p.add_argument('--sharpness', default=1.0, type=float,
                   help='Specify sharpness factor to use for weighted average.'
                   ' [%(default)s]')

    p.add_argument('--sigma', default=1.0, type=float,
                   help='Sigma of the gaussian to use. [%(default)s]')

    p.add_argument('--out_sym', action='store_true',
                   help='Save output in symmetric SH basis.')

    add_sh_basis_args(p)
    add_overwrite_arg(p)

    return p


def generate_mask(sh, threshold):
    norm = np.linalg.norm(sh, axis=-1)
    mask = norm > threshold
    return mask


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Checking args
    assert_outputs_exist(parser, args, [args.out_sh, args.out_mask])
    assert_inputs_exist(parser, args.in_sh)

    # Prepare data
    sh_img = nib.load(args.in_sh)
    data = sh_img.get_fdata(dtype=np.float64)

    logging.info('Executing locally asymmetric Gaussian filtering.')
    filtered_sh = local_asym_gaussian_filtering(
        data, sh_order=args.sh_order,
        sh_basis=args.sh_basis,
        out_full_basis=not(args.out_sym),
        sphere_str=args.sphere,
        dot_sharpness=args.sharpness,
        sigma=args.sigma)

    nib.save(nib.Nifti1Image(filtered_sh, sh_img.affine), args.out_sh)

    # Generate mask by applying threshold on input SH amplitude
    mask = generate_mask(data, args.mask_eps)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), sh_img.affine),
             args.out_mask)


if __name__ == "__main__":
    main()
