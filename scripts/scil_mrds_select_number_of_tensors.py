#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit the info from a NuFO map to the plausible MRDS solutions.

Each MRDS input is a list of 5 files:
    - Component size
    - Eigenvalues
    - Isotropic
    - Number of components
    - PDDs

    --N1 is the MRDS solution with 1 tensor.
    --N2 is the MRDS solution with 2 tensors.
    --N3 is the MRDS solution with 3 tensors.

    Example:
    scil_mrds_modsel_todi.py nufo.nii.gz
        --N1 V1_compsize.nii.gz
             V1_eigenvalues.nii.gz
             V1_isotropic.nii.gz
             V1_numcomp.nii.gz
             V1_pdds.nii.gz
        --N2 V2_compsize.nii.gz
             V2_eigenvalues.nii.gz
             V2_isotropic.nii.gz
             V2_numcomp.nii.gz
             V2_pdds.nii.gz
        --N3 V3_compsize.nii.gz
             V3_eigenvalues.nii.gz
             V3_isotropic.nii.gz
             V3_numcomp.nii.gz
             V3_pdds.nii.gz
"""

import argparse
import itertools
import logging

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_sh_basis_args, add_verbose_arg,
                             assert_headers_compatible,
                             assert_inputs_exist, assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_volume',
                   help='Volume with the number of expected tensors.'
                        ' (Example: NUFO volume)')

    g = p.add_argument_group(title='MRDS inputs')
    g.add_argument('--N1', nargs=5, required=True,
                   help='MRDS solution with 1 tensor.')
    g.add_argument('--N2', nargs=5, required=True,
                   help='MRDS solution with 2 tensors.')
    g.add_argument('--N3', nargs=5, required=True,
                   help='MRDS solution with 3 tensors.')

    p.add_argument('--prefix', default='results',
                   help='prefix of the MRDS results [%(default)s].')
    p.add_argument('--mask',
                   help='Optional mask filename.')

    add_processes_arg(p)
    add_sh_basis_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(args.verbose.upper())

    assert_inputs_exist(parser, [args.in_volume] + args.N1 + args.N2 + args.N3,
                        optional=args.mask)
    output_files = ["{}_MRDS_compsize.nii.gz".format(args.prefix),
                    "{}_MRDS_eigenvalues.nii.gz".format(args.prefix),
                    "{}_MRDS_isotropic.nii.gz".format(args.prefix),
                    "{}_MRDS_num_comp.nii.gz".format(args.prefix),
                    "{}_MRDS_pdds_cartesian.nii.gz".format(args.prefix)]
    assert_outputs_exist(parser, args, output_files)
    assert_headers_compatible(parser, [args.in_volume] +
                              args.N1 + args.N2 + args.N3)

    mrds_files = [args.N1, args.N2, args.N3]

    compsize = []
    eigenvalues = []
    iso = []
    numcomp = []
    pdds = []
    for N in range(3):
        compsize.append(nib.load(mrds_files[N][0]).get_fdata(dtype=np.float32))
        eigenvalues.append(nib.load(mrds_files[N][1]).get_fdata(dtype=np.float32))
        iso.append(nib.load(mrds_files[N][2]).get_fdata(dtype=np.float32))
        numcomp.append(nib.load(mrds_files[N][3]).get_fdata(dtype=np.float32))
        pdds.append(nib.load(mrds_files[N][4]).get_fdata(dtype=np.float32))

    # MOdel SElector MAP
    mosemap_img = nib.load(args.in_volume)
    mosemap = get_data_as_labels(mosemap_img)
    affine = mosemap_img.affine
    X, Y, Z = mosemap.shape[0:3]

    # load mask
    if args.mask:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
    else:
        mask = np.ones((X, Y, Z), dtype=np.uint8)

    # select data using mosemap
    voxels = itertools.product(range(X), range(Y), range(Z))
    filtered_voxels = ((x, y, z) for (x, y, z) in voxels if mask[x, y, z])

    compsize_out = np.zeros((X, Y, Z, 3))
    eigenvalues_out = np.zeros((X, Y, Z, 9))
    iso_out = np.zeros((X, Y, Z, 2))
    numcomp_out = np.zeros((X, Y, Z), dtype=np.uint8)
    pdds_out = np.zeros((X, Y, Z, 9))

    # select data using mosemap
    for (X, Y, Z) in filtered_voxels:
        N = mosemap[X, Y, Z]-1

        if N > 2:
            N = 2

        if N > -1:
            compsize_out[X, Y, Z, :] = compsize[N][X, Y, Z, :]
            eigenvalues_out[X, Y, Z, :] = eigenvalues[N][X, Y, Z, :]
            iso_out[X, Y, Z, :] = iso[N][X, Y, Z, :]
            numcomp_out[X, Y, Z] = int(numcomp[N][X, Y, Z])
            pdds_out[X, Y, Z, :] = pdds[N][X, Y, Z, :]

    # write output files
    nib.save(nib.Nifti1Image(compsize_out, affine), output_files[0])
    nib.save(nib.Nifti1Image(eigenvalues_out, affine), output_files[1])
    nib.save(nib.Nifti1Image(iso_out, affine), output_files[2])
    nib.save(nib.Nifti1Image(numcomp_out, affine), output_files[3])
    nib.save(nib.Nifti1Image(pdds_out, affine), output_files[4])


if __name__ == '__main__':
    main()
