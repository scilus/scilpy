#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to sample SF values from a Spherical Harmonics signal.
"""

import argparse

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.io import read_bvals_bvecs
from dipy.reconst.shm import sh_to_sf
import nibabel as nib
import numpy as np

from scilpy.gradientsampling.save_gradient_sampling import \
    save_gradient_sampling_fsl
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg,
                             add_sh_basis_args, assert_inputs_exist,
                             assert_outputs_exist)

ORDER_FROM_NCOEFFS = {1: 0, 6: 2, 15: 4, 28: 6, 45: 8}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh',
                   help='Path of the SH volume.')
    p.add_argument('in_dwi',
                   help='Path of the original DWI volume (to fetch b0 images).')
    p.add_argument('in_bval',
                   help='Path of the b-value file, in FSL format.')
    p.add_argument('in_bvec',
                   help='Path of the b-vector file, in FSL format.')

    p.add_argument('out_sf',
                   help='Name of the output SF file to save (bvals/bvecs will '
                        'be automatically named).')

    p.add_argument('--sphere', choices=['symmetric724', 'repulsion724'],
                   default='symmetric724',
                   help='Sphere to use for sampling SF. [%(default)s]')
    add_sh_basis_args(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_sh])

    out_bvecs = args.out_sf.replace(".nii.gz", ".bvec")
    out_bvals = args.out_sf.replace(".nii.gz", ".bval")

    assert_outputs_exist(parser, args, [args.out_sf, out_bvals, out_bvecs])

    # Load SH
    vol_sh = nib.load(args.in_sh)
    data_sh = vol_sh.get_fdata(dtype=np.float32)

    # Load DWI + gradients
    vol_dwi = nib.load(args.in_dwi)
    data_dwi = vol_dwi.get_fdata(dtype=np.float32)
    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    gtab = gradient_table(args.in_bval, args.in_bvec, b0_threshold=bvals.min())
    b0_volume = data_dwi[..., gtab.b0s_mask]

    # Figure out SH order
    n_coeffs = data_sh.shape[-1]
    sh_order = ORDER_FROM_NCOEFFS[n_coeffs]

    # Sample SF from SH
    sphere = get_sphere(args.sphere)
    sf = sh_to_sf(data_sh, sphere, sh_order=sh_order, basis_type=args.sh_basis)

    # Append b0 images
    sf = np.concatenate((sf, b0_volume), axis=-1)

    # Save SF
    nib.save(nib.Nifti1Image(sf.astype(np.float32), vol_sh.affine), args.out_sf)

    # Save new bvals/bvecs
    avg_bval = np.mean(gtab.bvals[np.logical_not(gtab.b0s_mask)])
    bvals = ([avg_bval] * len(sphere.theta)) + ([0] * b0_volume.shape[-1])
    bvecs = np.concatenate(
        (sphere.vertices, np.zeros((b0_volume.shape[-1], 3))), axis=0)
    save_gradient_sampling_fsl(bvecs, np.arange(len(bvals)), bvals,
                               out_bvals, out_bvecs)


if __name__ == "__main__":
    main()
