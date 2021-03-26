#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to sample SF values from a Spherical Harmonics signal. Outputs a Nifti
file with the SF values and an associated .bvec file with the chosen directions.

If converting from SH to a DWI-like SF volume, --in_bval and --in_b0 need
to be provided to concatenate the b0 image to the SF, and to generate the new
bvals file. Otherwise, no .bval file will be created.
"""

import argparse
from operator import xor

from dipy.data import SPHERE_FILES, get_sphere
from dipy.io import read_bvals_bvecs
from dipy.reconst.shm import order_from_ncoef, sh_to_sf
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_force_b0_arg, add_overwrite_arg,
                             add_sh_basis_args, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.bvec_bval_tools import (check_b0_threshold)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh',
                   help='Path of the SH volume.')
    p.add_argument('out_sf',
                   help='Name of the output SF file to save (bvals/bvecs will '
                        'be automatically named when necessary).')

    # Sphere choice for SF
    p.add_argument('--sphere', default='repulsion724',
                   choices=sorted(SPHERE_FILES.keys()),
                   help='Sphere used for the SH to SF projection. '
                        '[%(default)s]')

    # Optional subparser for a DWI-like volume
    sub = p.add_subparsers(help="Optional arguments to generate a "
                                "DWI-like output")
    dwi_subparser = sub.add_parser("dwi_like",
                                   description="Generate a DWI-like output, "
                                               "including a `.bval` file and "
                                               "b0 images in the sf file.")
    dwi_subparser.add_argument('--in_bval',
                               required=True,
                               help='b-value file, in FSL format, '
                                    'used to assign a b-value to the '
                                    'output SF and generate a `.bval` file.')
    dwi_subparser.add_argument('--in_b0',
                               required=True,
                               help='b0 volume to concatenate to the '
                                    'final SF volume.')
    add_sh_basis_args(p)

    add_overwrite_arg(p)
    add_force_b0_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_sh,
                        optional=[args.in_bval, args.in_b0])

    out_bvecs = args.out_sf.replace(".nii.gz", ".bvec")
    out_bvals = None
    if args.in_bval:
        out_bvals = args.out_sf.replace(".nii.gz", ".bval")

    assert_outputs_exist(parser, args, [args.out_sf, out_bvecs],
                         optional=out_bvals)

    # Either both --in_bval and --in_b0 are provided, or none
    if xor(bool(args.in_bval), bool(args.in_b0)):
        parser.error("If one of --in_bval or --in_b0 is provided, both "
                     "are required.")

    # Load SH
    vol_sh = nib.load(args.in_sh)
    data_sh = vol_sh.get_fdata(dtype=np.float32)

    # Figure out SH order
    sh_order = order_from_ncoef(data_sh.shape[-1], full_basis=False)

    # Sample SF from SH
    sphere = get_sphere(args.sphere)
    sf = sh_to_sf(data_sh, sphere, sh_order=sh_order, basis_type=args.sh_basis)
    new_bvecs = sphere.vertices

    if args.in_bval and args.in_b0:
        # Load b0
        vol_b0 = nib.load(args.in_b0)
        data_b0 = vol_b0.get_fdata(dtype=np.float32)
        if data_b0.ndim == 3:
            data_b0 = data_b0[..., np.newaxis]

        # Load bvals
        bvals, _ = read_bvals_bvecs(args.in_bval, None)

        # Compute average bval
        check_b0_threshold(args.force_b0_threshold, bvals.min())
        b0s_mask = bvals <= bvals.min()
        avg_bval = np.mean(bvals[np.logical_not(b0s_mask)])
        new_bvals = ([avg_bval] * len(sphere.theta)) + ([0] * data_b0.shape[-1])

        # Save new bvals
        np.savetxt(out_bvals, np.array(new_bvals)[None, :], fmt='%.3f')

        # Append zeros to bvecs
        new_bvecs = np.concatenate(
            (new_bvecs, np.zeros((data_b0.shape[-1], 3))), axis=0)

        # Append b0 images to SF
        sf = np.concatenate((sf, data_b0), axis=-1)

    # Save new bvecs
    np.savetxt(out_bvecs, new_bvecs.T, fmt='%.8f')

    # Save SF
    nib.save(nib.Nifti1Image(sf.astype(np.float32), vol_sh.affine), args.out_sf)


if __name__ == "__main__":
    main()
