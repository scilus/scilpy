#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to estimate asymmetric ODFs (aODFs) from a spherical harmonics image.

Two methods are available:
    * Unified filtering [1] combines four asymmetric filtering methods into
      a single equation and relies on a combination of four gaussian filters.
    * Cosine filtering [2] is a simpler implementation using cosine distance
      for assigning weights to neighbours.

Unified filtering can be accelerated using OpenCL with the option --use_opencl.
Make sure you have pyopencl installed before using this option. By default, the
OpenCL program will run on the cpu. To use a gpu instead, also specify the
option --device gpu.
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
                             assert_outputs_exist, parse_sh_basis_arg)
from scilpy.denoise.asym_filtering import (cosine_filtering, unified_filtering)


EPILOG = """
[1] Poirier and Descoteaux, 2024, "A Unified Filtering Method for Estimating
    Asymmetric Orientation Distribution Functions", Neuroimage, vol. 287,
    https://doi.org/10.1016/j.neuroimage.2024.120516

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

    p.add_argument('--sphere', default='repulsion200',
                   choices=sorted(SPHERE_FILES.keys()),
                   help='Sphere used for the SH to SF projection. '
                        '[%(default)s]')

    p.add_argument('--method', default='unified',
                   choices=['unified', 'cosine'],
                   help="Method for estimating asymmetric ODFs "
                        "[%(default)s].\nOne of:\n"
                        "    'unified': Unified filtering [1].\n"
                        "    'cosine' : Cosine-based filtering [2].")

    shared_group = p.add_argument_group('Shared filter arguments')
    shared_group.add_argument('--sigma_spatial', default=1.0, type=float,
                              help='Standard deviation for spatial distance.'
                                   ' [%(default)s]')

    unified_group = p.add_argument_group('Unified filter arguments')
    unified_group.add_argument('--sigma_align', default=0.8, type=float,
                               help='Standard deviation for alignment '
                                    'filter. [%(default)s]')
    unified_group.add_argument('--sigma_range', default=0.2, type=float,
                               help='Standard deviation for range filter\n'
                                    '*relative to SF range of image*. '
                                    '[%(default)s]')
    unified_group.add_argument('--sigma_angle', type=float,
                               help='Standard deviation for angular filter\n'
                                    '(disabled by default).')
    unified_group.add_argument('--disable_spatial', action='store_true',
                               help='Disable spatial filtering.')
    unified_group.add_argument('--disable_align', action='store_true',
                               help='Disable alignment filtering.')
    unified_group.add_argument('--disable_range', action='store_true',
                               help='Disable range filtering.')
    unified_group.add_argument('--include_center', action='store_true',
                               help='Include center voxel in neighourhood.')
    unified_group.add_argument('--win_hwidth', type=int,
                               help='Filtering window half-width. Defaults to '
                                    '3*sigma_spatial.')

    cosine_group = p.add_argument_group('Cosine filter arguments')
    cosine_group.add_argument('--sharpness', default=1.0, type=float,
                              help='Specify sharpness factor to use for\n'
                                   'weighted average. [%(default)s]')

    p.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                   help='Device to use for execution. [%(default)s]')
    p.add_argument('--use_opencl', action='store_true',
                   help='Accelerate code using OpenCL (requires pyopencl\n'
                        'and a working OpenCL implementation).')
    p.add_argument('--patch_size', type=int, default=40,
                   help='OpenCL patch size. [%(default)s]')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if args.device == 'gpu' and not args.use_opencl:
        logging.warning("Device 'gpu' chosen but --use_opencl not specified. "
                        "Proceeding with use_opencl=True.")
        # force use_opencl if option gpu is chosen
        args.use_opencl = True

    if args.use_opencl and args.method == 'cosine':
        parser.error('Option --use_opencl is not supported'
                     ' for cosine filtering.')

    outputs = [args.out_sh]
    if args.out_sym:
        outputs.append(args.out_sym)
    assert_outputs_exist(parser, args, outputs)
    assert_inputs_exist(parser, args.in_sh)

    # Prepare data
    sh_img = nib.load(args.in_sh)
    data = sh_img.get_fdata(dtype=np.float32)

    sh_order, full_basis = get_sh_order_and_fullness(data.shape[-1])
    sh_basis, is_legacy = parse_sh_basis_arg(args)

    t0 = time.perf_counter()
    logging.info('Filtering SH image.')
    if args.method == 'unified':
        sigma_align = None if args.disable_align else args.sigma_align
        sigma_range = None if args.disable_range else args.sigma_range
        sigma_spatial = None if args.disable_spatial else args.sigma_spatial

        asym_sh = unified_filtering(
            data, sh_order=sh_order, sh_basis=sh_basis,
            is_legacy=is_legacy, full_basis=full_basis,
            sphere_str=args.sphere,
            sigma_spatial=sigma_spatial,
            sigma_align=sigma_align,
            sigma_angle=args.sigma_angle,
            rel_sigma_range=sigma_range,
            win_hwidth=args.win_hwidth,
            exclude_center=not args.include_center,
            device_type=args.device,
            use_opencl=args.use_opencl)
    else:  # args.method == 'cosine'
        asym_sh = cosine_filtering(
            data, sh_order=sh_order,
            sh_basis=sh_basis,
            in_full_basis=full_basis,
            is_legacy=is_legacy,
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
