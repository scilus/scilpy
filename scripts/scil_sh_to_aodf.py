#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to estimate asymmetric ODFs (aODFs) from a spherical harmonics image.

Two methods are available:
    * Angle-aware bilateral filtering [1] is an extension of bilateral
      filtering considering the angular distance between sphere directions
      for filtering 5-dimensional spatio-angular images.
    * Cosine filtering [2] is a simpler implementation using cosine distance
      for assigning weights to neighbours.

Angle-aware bilateral filtering can be performed on the GPU using pyopencl by
specifying --use_gpu. Make sure you have pyopencl installed to use this option.
Otherwise, the filtering runs entirely on the CPU.
"""

import argparse
import logging
import time
import nibabel as nib
import numpy as np

from dipy.data import SPHERE_FILES
from dipy.reconst.shm import sph_harm_ind_list
from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, add_sh_basis_args,
                             assert_outputs_exist)
from scilpy.denoise.asym_filtering import (cosine_filtering,
                                           angle_aware_bilateral_filtering)


EPILOG = """
[1] Poirier et al, 2022, "Intuitive Angle-Aware Bilateral Filtering Revealing
    Asymmetric Fiber ODF for Improved Tractography", ISMRM 2022 (abstract 3552)

[2] Poirier et al, 2021, "Investigating the Occurrence of Asymmetric Patterns
    in White Matter Fiber Orientation Distribution Functions", ISMRM 2021
    (abstract 0865)
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh',
                   help='Path to the input file.')

    p.add_argument('out_sh',
                   help='File name for averaged signal.')

    p.add_argument('--out_sym', default=None,
                   help='Name of optional symmetric output. [%(default)s]')

    add_sh_basis_args(p)

    p.add_argument('--sphere', default='repulsion724',
                   choices=sorted(SPHERE_FILES.keys()),
                   help='Sphere used for the SH to SF projection. '
                        '[%(default)s]')

    p.add_argument('--method', default='bilateral',
                   choices=['bilateral', 'cosine'],
                   help='Method for estimating asymmetric ODFs '
                        '[%(default)s].\nOne of:\n'
                        '    \'bilateral\': Angle-aware bilateral '
                        'filtering [1].\n'
                        '    \'cosine\'  : Cosine-based filtering [2].')

    shared_group = p.add_argument_group('Shared filter arguments')
    shared_group.add_argument('--sigma_spatial', default=1.0, type=float,
                              help='Standard deviation for spatial distance.'
                                   ' [%(default)s]')

    trilateral_group = p.add_argument_group('Angle-aware bilateral arguments')
    trilateral_group.add_argument('--sigma_angular', default=1.0, type=float,
                                  help='Standard deviation for angular '
                                       'distance. [%(default)s]')
    trilateral_group.add_argument('--sigma_range', default=1.0, type=float,
                                  help='Standard deviation for range filter.'
                                       ' [%(default)s]')

    cosine_group = p.add_argument_group('Cosine filter arguments')
    cosine_group.add_argument('--sharpness', default=1.0, type=float,
                              help='Specify sharpness factor to use for'
                                   ' weighted average. [%(default)s]')

    p.add_argument('--use_gpu', action='store_true',
                   help='Use GPU for computation.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    if args.use_gpu and args.method == 'cosine':
        parser.error('Option --use_gpu is not supported for cosine filtering.')

    outputs = [args.out_sh]
    if args.out_sym:
        outputs.append(args.out_sym)
    assert_outputs_exist(parser, args, outputs)
    assert_inputs_exist(parser, args.in_sh)

    # Prepare data
    sh_img = nib.load(args.in_sh)
    data = sh_img.get_fdata(dtype=np.float32)

    sh_order, full_basis = get_sh_order_and_fullness(data.shape[-1])

    t0 = time.perf_counter()
    logging.info('Filtering SH image.')
    if args.method == 'bilateral':
        asym_sh = angle_aware_bilateral_filtering(
            data, sh_order=sh_order,
            sh_basis=args.sh_basis,
            in_full_basis=full_basis,
            sphere_str=args.sphere,
            sigma_spatial=args.sigma_spatial,
            sigma_angular=args.sigma_angular,
            sigma_range=args.sigma_range,
            use_gpu=args.use_gpu)
    else:  # args.method == 'cosine'
        asym_sh = cosine_filtering(
            data, sh_order=sh_order,
            sh_basis=args.sh_basis,
            in_full_basis=full_basis,
            sphere_str=args.sphere,
            dot_sharpness=args.sharpness,
            sigma=args.sigma_spatial)

    t1 = time.perf_counter()
    logging.info('Elapsed time (s): {0}'.format(t1 - t0))

    logging.info('Saving filtered SH to file {0}.'.format(args.out_sh))
    nib.save(nib.Nifti1Image(asym_sh, sh_img.affine), args.out_sh)

    if args.out_sym:
        _, orders = sph_harm_ind_list(sh_order, full_basis=True)
        logging.info('Saving symmetric SH to file {0}.'.format(args.out_sym))
        nib.save(nib.Nifti1Image(asym_sh[..., orders % 2 == 0], sh_img.affine),
                 args.out_sym)


if __name__ == '__main__':
    main()
