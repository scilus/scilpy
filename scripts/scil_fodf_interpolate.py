#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interpolate a spherical harmonics (SH) volume to a different resolution
by projecting the SH from its riemmanian manifold to the Euclidian space
and back.

Reference: Cheng, J., Ghosh, A., Jiang, T., & Deriche, R. (2009, September). A Riemannian framework for orientation distribution function computing. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 911-918). Berlin, Heidelberg: Springer Berlin Heidelberg.

"""

import argparse
import logging

import nibabel as nib
import numpy as np
from dipy.data import SPHERE_FILES, get_sphere

from scilpy.io.utils import (assert_inputs_exist,
                             add_overwrite_arg, add_processes_arg,
                             add_sh_basis_args,
                             add_verbose_arg,
                             assert_outputs_exist,
                             parse_sh_basis_arg,
                             validate_nbr_processes)
from scilpy.reconst.utils import get_sh_order_and_fullness

from scilpy.reconst.fodf import interpolate_fodf


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh',
                   help='Path of the SH volume to interpolate.')
    p.add_argument('out_sh',
                   help='Path of the output SH volume.')
    p.add_argument('resolution', type=int, nargs='+',
                   help='Resolution of the output SH volume.')

    # Sphere vs bvecs choice for SF
    p.add_argument('--sphere', choices=sorted(SPHERE_FILES.keys()),
                   help='Sphere used for the SH to SF projection. ')

    add_sh_basis_args(p)
    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_sh)
    assert_outputs_exist(parser, args, args.out_sh)

    nbr_processes = validate_nbr_processes(parser, args)
    sh_basis, is_legacy = parse_sh_basis_arg(args)

    # Load SH
    vol_sh = nib.load(args.in_sh)
    data_sh = vol_sh.get_fdata(dtype=np.float32)

    K = data_sh.shape[-1]

    sh_order, fullness = get_sh_order_and_fullness(K)

    # Sample SF from SH
    sphere = get_sphere(args.sphere)

    # Interpolate SH
    sh_pow = interpolate_fodf(
        data_sh, args.resolution, sphere, sh_order, fullness, sh_basis,
        is_legacy, nbr_processes)

    # Save SH
    nib.save(nib.Nifti1Image(sh_pow, vol_sh.affine), args.out_sh)


if __name__ == "__main__":
    main()
