#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute angle-aware bilateral filtering.

Angle-aware bilateral filtering is an extension of bilateral filtering
considering the angular distance between sphere directions for filtering
5-dimensional spatio-angular images.

The filtering can be performed on the GPU using pyopencl by specifying
--use_gpu. Make sure you have pyopencl installed to use this option.
Otherwise, the filtering also runs entirely on the CPU, optionally using
multiple processes.

Using default parameters, fODF filtering for a HCP subject processed with
Tractoflow takes about 12 minutes on the GPU versus 90 minutes using 16 CPU
threads. The time required scales with the sigma_spatial parameter. For
example, sigma_spatial=3.0 takes about 4.15 hours on the GPU versus 7.67 hours
on the CPU using 16 threads.
"""

import argparse
import logging
import time
import nibabel as nib
import numpy as np

from dipy.data import SPHERE_FILES
from dipy.reconst.shm import sph_harm_ind_list
from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             add_sh_basis_args,
                             assert_outputs_exist,
                             validate_nbr_processes)
from scilpy.denoise.bilateral_filtering import angle_aware_bilateral_filtering


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_sh',
                   help='Path to the input file.')

    p.add_argument('out_sh',
                   help='File name for averaged signal.')

    add_sh_basis_args(p)

    p.add_argument('--out_sym', default=None,
                   help='Name of optional symmetric output. [%(default)s]')

    p.add_argument('--sphere', default='repulsion724',
                   choices=sorted(SPHERE_FILES.keys()),
                   help='Sphere used for the SH to SF projection. '
                        '[%(default)s]')

    p.add_argument('--sigma_angular', default=1.0, type=float,
                   help='Standard deviation for angular distance.'
                        ' [%(default)s]')

    p.add_argument('--sigma_spatial', default=1.0, type=float,
                   help='Standard deviation for spatial distance.'
                        ' [%(default)s]')

    p.add_argument('--sigma_range', default=1.0, type=float,
                   help='Standard deviation for range filter.'
                        ' [%(default)s]')

    p.add_argument('--use_gpu', action='store_true',
                   help='Use GPU for computation.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_processes_arg(p)

    return p


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

    nbr_processes = validate_nbr_processes(parser, args)

    # Prepare data
    sh_img = nib.load(args.in_sh)
    data = sh_img.get_fdata(dtype=np.float32)

    sh_order, full_basis = get_sh_order_and_fullness(data.shape[-1])

    t0 = time.perf_counter()
    logging.info('Executing angle-aware bilateral filtering.')
    asym_sh = angle_aware_bilateral_filtering(
        data, sh_order=sh_order,
        sh_basis=args.sh_basis,
        in_full_basis=full_basis,
        sphere_str=args.sphere,
        sigma_spatial=args.sigma_spatial,
        sigma_angular=args.sigma_angular,
        sigma_range=args.sigma_range,
        use_gpu=args.use_gpu,
        nbr_processes=nbr_processes)
    t1 = time.perf_counter()
    logging.info('Elapsed time (s): {0}'.format(t1 - t0))

    logging.info('Saving filtered SH to file {0}.'.format(args.out_sh))
    nib.save(nib.Nifti1Image(asym_sh, sh_img.affine), args.out_sh)

    if args.out_sym:
        _, orders = sph_harm_ind_list(sh_order, full_basis=True)
        logging.info('Saving symmetric SH to file {0}.'.format(args.out_sym))
        nib.save(nib.Nifti1Image(asym_sh[..., orders % 2 == 0], sh_img.affine),
                 args.out_sym)


if __name__ == "__main__":
    main()
