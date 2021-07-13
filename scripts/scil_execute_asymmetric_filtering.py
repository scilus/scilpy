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

import nibabel as nib
import numpy as np

from dipy.data import SPHERE_FILES
from dipy.reconst.shm import order_from_ncoef, sph_harm_ind_list

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

    p.add_argument('--out_asym_map', default='asym_map.nii.gz',
                   help='File name for asymmetry map (Cetin Karayumak et al).'
                        '\nCan only be outputed when the output SH basis is '
                        'full. [%(default)s]')

    p.add_argument('--out_oddpwr_map', default='oddpwr_map.nii.gz',
                   help='File name for odd power map.\nWill only be outputed'
                        ' when output SH basis is full. [%(default)s]')

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

    p.add_argument('--edge_mode', default='same',
                   choices=['same', 'wall'],
                   help='Specify how edges are processed.\n'
                        '    \'same\': Edges are processed in the same way as'
                        ' the rest of the image;\n'
                        '    \'wall\': Background voxels are discarded from '
                        'the average. The filter is\n            '
                        'updated and normalized for each voxel. Requires '
                        '\'--mask\' or\n            \'--sh0_th\'.'
                        ' [%(default)s]')

    mask_group = p.add_mutually_exclusive_group()
    mask_group.add_argument('--mask',
                            help='Mask when edge_mode is \'wall\'.')
    mask_group.add_argument('--sh0_th', type=float,
                            help='Threshold on SH0 coefficient '
                                 'when edge_mode is \'wall\'.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def compute_karayumak_asym_map(sh_coeffs):
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


def compute_odd_power_map(sh_coeffs):
    order = order_from_ncoef(sh_coeffs.shape[-1], full_basis=True)
    _, l_list = sph_harm_ind_list(order, full_basis=True)
    odd_l_list = (l_list % 2 == 1).reshape((1, 1, 1, -1))

    odd_order_norm = np.linalg.norm(
        sh_coeffs * odd_l_list,
        ord=2,
        axis=-1)

    full_order_norm = np.linalg.norm(
        sh_coeffs,
        ord=2,
        axis=-1)

    asym_map = np.zeros(sh_coeffs.shape[:-1])
    mask = full_order_norm > 0
    asym_map[mask] = odd_order_norm[mask] / full_order_norm[mask]

    return asym_map


def _assert_edge_mode(parser, args):
    if args.edge_mode == 'same':
        if args.mask is not None:
            parser.error('Cannot specify mask with edge_mode \'same\'.')
        if args.sh0_th is not None:
            parser.error('Cannot specify sh0_th with edge_mode \'same\'.')
    elif args.edge_mode == 'wall':
        if args.mask is None and args.sh0_th is None:
            parser.error('Missing required \'mask\' or \'sh0_th\' '
                         'argument for edge_mode \'wall\'.')


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    outputs = [args.out_sh]
    if not args.out_sym:
        outputs.append(args.out_asym_map)
        outputs.append(args.out_oddpwr_map)

    _assert_edge_mode(parser, args)
    inputs = [args.in_sh]
    if args.mask:
        inputs.append(args.mask)

    # Checking args
    assert_outputs_exist(parser, args, outputs)
    assert_inputs_exist(parser, inputs)

    # Prepare data
    sh_img = nib.load(args.in_sh)
    data = sh_img.get_fdata(dtype=np.float32)

    mask = None
    if args.edge_mode == 'wall':
        if args.mask:
            mask = get_data_as_mask(nib.load(args.mask), bool)
        else:
            mask = data[..., 0] > args.sh0_th

    logging.info('Executing locally asymmetric Gaussian filtering.')
    filtered_sh = local_asym_gaussian_filtering(
        data, sh_order=args.sh_order,
        sh_basis=args.sh_basis,
        out_full_basis=not(args.out_sym),
        sphere_str=args.sphere,
        dot_sharpness=args.sharpness,
        sigma=args.sigma,
        mask=mask)

    logging.info('Saving filtered SH to file {0}.'.format(args.out_sh))
    nib.save(nib.Nifti1Image(filtered_sh, sh_img.affine), args.out_sh)

    # Save asymmetry measure map when the output is in full SH basis
    if args.out_sym:
        logging.info('Skipping asymmetry map because output is symmetric.')
    else:
        logging.info('Generating asymmetry map from output.')
        asym_map = compute_karayumak_asym_map(filtered_sh)
        logging.info('Saving asymmetry map to file '
                     '{0}.'.format(args.out_asym_map))
        nib.save(nib.Nifti1Image(asym_map, sh_img.affine), args.out_asym_map)

        logging.info('Generating odd power map from output.')
        asym_map = compute_odd_power_map(filtered_sh)
        logging.info('Saving asymmetry map to file '
                     '{0}.'.format(args.out_oddpwr_map))
        nib.save(nib.Nifti1Image(asym_map, sh_img.affine), args.out_oddpwr_map)


if __name__ == "__main__":
    main()
