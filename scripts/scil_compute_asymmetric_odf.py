#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute asymmetric ODF by filtering an input symmetric ODF image.
The filtering is accelerated with numba and opencl for python.

The script supports any spherical signal expressed as a series of spherical
harmonics (SH) coefficients. For an explanation of the available SH bases, refer
to the DIPY documentation (https://dipy.org/documentation/1.4.0./theory/sh_basis/).
"""
import argparse
import logging
import time
import nibabel as nib
import numpy as np

from dipy.data import SPHERE_FILES, get_sphere
from dipy.reconst.shm import sph_harm_ind_list, sh_to_sf
from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist
from scilpy.denoise.unified_asym import AsymmetricFilter


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_sh',
                   help='Path to the input file.')

    p.add_argument('out_sh',
                   help='File name for averaged signal.')

    p.add_argument('--sh_basis', default='descoteaux07_legacy',
                   choices=['tournier07', 'descoteaux07',
                            'tournier07_legacy', 'descoteaux07_legacy'],
                   help='SH basis used for signal representation.'
                        ' [%(default)s]')

    p.add_argument('--out_sym', default=None,
                   help='Name of optional symmetric output. [%(default)s]')

    p.add_argument('--sphere', default='repulsion200',
                   choices=sorted(SPHERE_FILES.keys()),
                   help='Sphere used for the SH to SF projection. '
                        '[%(default)s]')

    p.add_argument('--sigma_spatial', default=1.0, type=float,
                   help='Standard deviation for spatial regularizer.'
                        ' [%(default)s]')

    p.add_argument('--sigma_align', default=0.8, type=float,
                   help='Standard deviation for alignment regularizer.'
                        ' [%(default)s]')

    p.add_argument('--sigma_angle', default=0.1, type=float,
                   help='Standard deviation for angular regularizer.'
                        ' [%(default)s]')

    p.add_argument('--sigma_range', default=0.2, type=float,
                   help='Standard deviation for range regularizer\n'
                        '**given as a ratio of the range of SF amplitudes \n'
                        'in the image**. [%(default)s]')

    p.add_argument('--exclude_self', action='store_true',
                   help='Exclude current voxel from neighbours.')

    p.add_argument('--disable_spatial', action='store_true',
                   help='Disable spatial filter.')

    p.add_argument('--disable_align', action='store_true',
                   help='Disable align filter.')

    p.add_argument('--disable_angle', action='store_true',
                   help='Disable angle filter.')

    p.add_argument('--disable_range', action='store_true',
                   help='Disable range filter.')

    p.add_argument('-v', action='store_true', dest='verbose',
                   help='If set, produces verbose output.')

    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')

    return p


def _parse_sh_basis(sh_basis_name):
    basis = 'descoteaux07' if 'descoteaux07' in sh_basis_name else 'tournier07'
    legacy = 'legacy' in sh_basis_name
    return basis, legacy


def get_sf_range(data, sh_order, full_basis, sphere_name):
    sphere = get_sphere(sphere_name)
    sf = np.array([sh_to_sf(sh, sphere, sh_order, full_basis=full_basis)
                   for sh in data], dtype=np.float32)
    vrange = np.max(sf) - np.min(sf)
    return vrange


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Checking args
    outputs = [args.out_sh]
    if args.out_sym:
        outputs.append(args.out_sym)
    assert_outputs_exist(parser, args, outputs)
    assert_inputs_exist(parser, args.in_sh)

    # Prepare data
    sh_img = nib.load(args.in_sh)
    data = sh_img.get_fdata(dtype=np.float32)
    sh_order, full_basis = get_sh_order_and_fullness(data.shape[-1])

    sh_basis, legacy = _parse_sh_basis(args.sh_basis)

    sigma_range = args.sigma_range
    if not args.disable_range:
        vrange = get_sf_range(data, sh_order, full_basis, args.sphere)
        sigma_range = args.sigma_range * vrange

    t0 = time.perf_counter()
    logging.info('Executing asymmetric filtering.')
    asym_filter = AsymmetricFilter(sh_order, sh_basis, legacy,
                                   full_basis, sphere_str=args.sphere,
                                   sigma_spatial=args.sigma_spatial,
                                   sigma_align=args.sigma_align,
                                   sigma_angle=args.sigma_angle,
                                   sigma_range=sigma_range,
                                   exclude_self=args.exclude_self,
                                   disable_spatial=args.disable_spatial,
                                   disable_align=args.disable_align,
                                   disable_angle=args.disable_angle,
                                   disable_range=args.disable_range)
    asym_sh = asym_filter(data)
    t1 = time.perf_counter()
    logging.info('Elapsed time (s): {0}'.format(t1 - t0))

    logging.info('Saving filtered SH to file {0}.'.format(args.out_sh))
    nib.save(nib.Nifti1Image(asym_sh.astype(np.float32), sh_img.affine), args.out_sh)

    if args.out_sym:
        _, orders = sph_harm_ind_list(sh_order, full_basis=True)
        logging.info('Saving symmetric SH to file {0}.'.format(args.out_sym))
        nib.save(nib.Nifti1Image(asym_sh[..., orders % 2 == 0].astype(np.float32), sh_img.affine),
                 args.out_sym)


if __name__ == "__main__":
    main()
