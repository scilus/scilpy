#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to sample SF values from a Spherical Harmonics signal. Outputs a Nifti
file with the SF values and an associated .bvec file with the chosen
directions.

If converting from SH to a DWI-like SF volume, --in_bval and --in_b0 need
to be provided to concatenate the b0 image to the SF, and to generate the new
bvals file. Otherwise, no .bval file will be created.

Formerly: scil_compute_sf_from_sh.py
"""

import argparse

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere
from dipy.data import SPHERE_FILES, get_sphere
from dipy.io import read_bvals_bvecs

from scilpy.io.utils import (add_force_b0_arg, add_overwrite_arg,
                             add_processes_arg, add_sh_basis_args,
                             assert_inputs_exist, add_verbose_arg,
                             assert_outputs_exist, validate_nbr_processes)
from scilpy.reconst.sh import convert_sh_to_sf
from scilpy.gradients.bvec_bval_tools import (check_b0_threshold)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh',
                   help='Path of the SH volume.')
    p.add_argument('out_sf',
                   help='Name of the output SF file to save (bvals/bvecs will '
                        'be automatically named when necessary).')

    # Sphere vs bvecs choice for SF
    directions = p.add_mutually_exclusive_group(required=True)
    directions.add_argument('--sphere',
                            choices=sorted(SPHERE_FILES.keys()),
                            help='Sphere used for the SH to SF projection. ')
    directions.add_argument('--in_bvec',
                            help="Directions used for the SH to SF "
                            "projection.")

    p.add_argument('--dtype', default="float32",
                   choices=["float32", "float64"],
                   help="Datatype to use for SF computation and output array."
                        "'[%(default)s]'")

    # Optional args for a DWI-like volume
    p.add_argument('--in_bval',
                   help='b-value file, in FSL format, '
                        'used to assign a b-value to the '
                        'output SF and generate a `.bval` file.')
    p.add_argument('--in_b0',
                   help='b0 volume to concatenate to the '
                        'final SF volume.')
    p.add_argument('--out_bval',
                   help="Optional output bval file.")
    p.add_argument('--out_bvec',
                   help="Optional output bvec file.")

    p.add_argument('--b0_scaling', action="store_true",
                   help="Scale resulting SF by the b0 image.")

    add_sh_basis_args(p)
    p.add_argument('--full_basis', action="store_true",
                   help="If true, use a full basis for the input SH "
                        "coefficients.")

    add_processes_arg(p)
    add_verbose_arg(p)
    add_force_b0_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_sh,
                        optional=[args.in_bvec, args.in_bval, args.in_b0])
    assert_outputs_exist(parser, args, args.out_sf,
                         optional=[args.out_bvec, args.out_bval])

    if (args.in_bval and not args.out_bval) or (
            args.out_bval and not args.in_bval):
        parser.error("--out_bval is required if --in_bval is provided, "
                     "and vice-versa.")

    if args.in_bvec and not args.in_bval:
        parser.error(
            "--in_bval is required when using --in_bvec, in order to remove "
            "bvecs corresponding to b0 images.")

    if args.b0_scaling and not args.in_b0:
        parser.error("--in_b0 is required when using --b0_scaling.")

    nbr_processes = validate_nbr_processes(parser, args)

    # Load SH
    vol_sh = nib.load(args.in_sh)
    data_sh = vol_sh.get_fdata(dtype=np.float32)

    # Sample SF from SH
    if args.sphere:
        sphere = get_sphere(args.sphere)
    elif args.in_bvec:
        bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
        gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())
        # Remove bvecs corresponding to b0 images
        bvecs = bvecs[np.logical_not(gtab.b0s_mask)]
        sphere = Sphere(xyz=bvecs)

    sf = convert_sh_to_sf(data_sh, sphere,
                          input_basis=args.sh_basis,
                          input_full_basis=args.full_basis,
                          dtype=args.dtype,
                          nbr_processes=nbr_processes)
    new_bvecs = sphere.vertices.astype(np.float32)

    # Assign bval to SF if --in_bval was provided
    new_bvals = []
    if args.in_bval:
        # Load bvals
        bvals, _ = read_bvals_bvecs(args.in_bval, None)

        # Compute average bval
        b0_thr = check_b0_threshold(
            args.force_b0_threshold, bvals.min(), bvals.min())
        b0s_mask = bvals <= b0_thr
        avg_bval = np.mean(bvals[np.logical_not(b0s_mask)])

        new_bvals = ([avg_bval] * len(sphere.theta))

    # Add b0 images to SF (and bvals if necessary) if --in_b0 was provided
    if args.in_b0:
        # Load b0
        vol_b0 = nib.load(args.in_b0)
        data_b0 = vol_b0.get_fdata(dtype=args.dtype)
        if data_b0.ndim == 3:
            data_b0 = data_b0[..., np.newaxis]

        new_bvals = ([0] * data_b0.shape[-1]) + new_bvals

        # Append zeros to bvecs
        new_bvecs = np.concatenate(
            (np.zeros((data_b0.shape[-1], 3)), new_bvecs), axis=0)

        # Scale SF by b0
        if args.b0_scaling:
            # Clip SF signal between 0. and 1., then scale using mean b0
            sf = np.clip(sf, 0., 1.)
            scale_b0 = np.mean(data_b0, axis=-1, keepdims=True)
            sf = sf * scale_b0

        # Append b0 images to SF
        sf = np.concatenate((data_b0, sf), axis=-1)

    # Save new bvals
    if args.out_bval:
        np.savetxt(args.out_bval, np.array(new_bvals)[None, :], fmt='%.3f')

    # Save new bvecs
    if args.out_bvec:
        np.savetxt(args.out_bvec, new_bvecs.T, fmt='%.8f')

    # Save SF
    nib.save(nib.Nifti1Image(sf, vol_sh.affine), args.out_sf)


if __name__ == "__main__":
    main()
