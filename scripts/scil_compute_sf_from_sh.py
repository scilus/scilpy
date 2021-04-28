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

from dipy.data import SPHERE_FILES, get_sphere
from dipy.io import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_force_b0_arg, add_overwrite_arg,
                             add_processes_arg, add_sh_basis_args,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.reconst.multi_processes import convert_sh_to_sf
from scilpy.utils.bvec_bval_tools import (check_b0_threshold)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh',
                   help='Path of the SH volume.')
    p.add_argument('out_sf',
                   help='Name of the output SF file to save (bvals/bvecs will '
                        'be automatically named when necessary).')

    p.add_argument('--full_basis', action="store_true",
                   help="If true, use a full basis for the input SH "
                        "coefficients.")

    # Sphere choice for SF
    p.add_argument('--sphere', default='repulsion724',
                   choices=sorted(SPHERE_FILES.keys()),
                   help='Sphere used for the SH to SF projection. '
                        '[%(default)s]')
    p.add_argument('--dtype', default="float32",
                   choices=["float32", "float64"],
                   help="Datatype to use for SF computation and output array."
                        "'[%(default)s]'")

    # Optional args for a DWI-like volume
    p.add_argument("--extract_as_dwi", action="store_true",
                   help="Generate a DWI-like output, including a `.bval` file "
                        "and b0 images in the sf file.")
    p.add_argument('--bval',
                   help='b-value file, in FSL format, '
                        'used to assign a b-value to the '
                        'output SF and generate a `.bval` file.')
    p.add_argument('--b0',
                   help='b0 volume to concatenate to the '
                        'final SF volume.')
    add_sh_basis_args(p)
    add_processes_arg(p)

    add_overwrite_arg(p)
    add_force_b0_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_sh)

    out_bvecs = args.out_sf.replace(".nii.gz", ".bvec")
    assert_outputs_exist(parser, args, [args.out_sf, out_bvecs])

    if args.extract_as_dwi:
        assert_inputs_exist(parser, [args.bval, args.b0])
        out_bvals = args.out_sf.replace(".nii.gz", ".bval")
        assert_outputs_exist(parser, args, out_bvals)
    else:
        if args.bval or args.b0:
            parser.error("--bval and --b0 are required only when "
                         "--extract_as_dwi is used.")

    # Load SH
    vol_sh = nib.load(args.in_sh)
    data_sh = vol_sh.get_fdata(dtype=np.float32)

    # Sample SF from SH
    sphere = get_sphere(args.sphere)
    sf = convert_sh_to_sf(data_sh, sphere,
                          input_basis=args.sh_basis,
                          input_full_basis=args.full_basis,
                          dtype=args.dtype,
                          nbr_processes=args.nbr_processes)
    new_bvecs = sphere.vertices.astype(np.float32)

    if args.extract_as_dwi:
        # Load b0
        vol_b0 = nib.load(args.b0)
        data_b0 = vol_b0.get_fdata(dtype=np.float32)
        if data_b0.ndim == 3:
            data_b0 = data_b0[..., np.newaxis]

        # Load bvals
        bvals, _ = read_bvals_bvecs(args.bval, None)

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
