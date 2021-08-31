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
is used, asymmetry maps are generated using an asymmetry measure from Cetin
Karayumak et al, 2018, and our own asymmetry measure defined as the ratio of
the L2-norm of odd SH coefficients on the L2-norm of all SH coefficients.

Using default parameters, the script completes in about 15-20 minutes for a
HCP subject fiber ODF processed with tractoflow. Also note the bigger the
sphere used for SH to SF projection, the higher the RAM consumption and
compute time.
"""

import argparse
import logging
from scilpy.reconst.utils import get_sh_order_and_fullness

import nibabel as nib
import numpy as np

from dipy.data import SPHERE_FILES

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             add_sh_basis_args,
                             assert_outputs_exist)

from scilpy.io.image import get_data_as_mask

from scilpy.denoise.asym_enhancement import local_asym_gaussian_filtering


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_sh',
                   help='Path to the input file.')

    p.add_argument('out_sh',
                   help='File name for averaged signal.')

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


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Checking args
    assert_outputs_exist(parser, args, args.out_sh)
    assert_inputs_exist(parser, args.in_sh)

    # Prepare data
    sh_img = nib.load(args.in_sh)
    data = sh_img.get_fdata(dtype=np.float32)

    sh_order, full_basis = get_sh_order_and_fullness(data.shape[-1])

    logging.info('Executing locally asymmetric Gaussian filtering.')
    filtered_sh = local_asym_gaussian_filtering(
        data, sh_order=sh_order,
        sh_basis=args.sh_basis,
        in_full_basis=full_basis,
        out_full_basis=not(args.out_sym),
        sphere_str=args.sphere,
        dot_sharpness=args.sharpness,
        sigma=args.sigma)

    logging.info('Saving filtered SH to file {0}.'.format(args.out_sh))
    nib.save(nib.Nifti1Image(filtered_sh, sh_img.affine), args.out_sh)


if __name__ == "__main__":
    main()
