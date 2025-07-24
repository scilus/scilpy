#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use the NUFO map information to select the plausible number of tensors
in the Multi-Resolution Discrete Search (MRDS).
https://link.springer.com/chapter/10.1007/978-3-031-47292-3_4

scil_mrds_select_number_of_tensors.py uses the output from mdtmrds command.
Some mdtmrds output files will be named differently from the expected input:
    COMP_SIZE becomes signal_fraction
    NUM_COMP becomes num_tensors
    PDDs_CARTESIAN becomes evecs
    Eigenvalues becomes evals

mdtmrds: information available soon (not part of scilpy).

Input:
    Inputs are a list of 5 files for each MRDS solution (D1, D2, D3).
    - Signal fraction of each tensor ([in_prefix]_D[1,2,3]_signal_fraction.nii.gz)
    - Eigenvalues ($in_prefix]_D[1,2,3]_evals.nii.gz)
    - Isotropic ([in_prefix]_D[1,2,3]_isotropic.nii.gz)
    - Number of tensors ([in_prefix]_D[1,2,3]_num_tensors.nii.gz)
    - Eigenvectors ([in_prefix]_D[1,2,3]_evecs.nii.gz)


    Example:
        scil_mrds_select_number_of_tensors.py sub-01 nufo.nii.gz
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
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)
    p.add_argument('in_prefix',
                   help='Prefix used for all MRDS solutions.')
    p.add_argument('in_volume',
                   help='Volume with the number of expected tensors.'
                        ' (Example: NUFO volume)')

    p.add_argument('--out_prefix', default='results',
                   help='Prefix of the MRDS results [%(default)s].')
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

    mrds_files = []
    for i in range(1, 4):
        mrds_files.append([args.in_prefix + '_D{}_signal_fraction.nii.gz'.format(i),
                           args.in_prefix + '_D{}_evals.nii.gz'.format(i),
                           args.in_prefix + '_D{}_isotropic.nii.gz'.format(i),
                           args.in_prefix + '_D{}_num_tensors.nii.gz'.format(i),
                           args.in_prefix + '_D{}_evecs.nii.gz'.format(i)])

    assert_inputs_exist(parser, [args.in_volume] + [x for xs in mrds_files for x in xs],
                        optional=args.mask)

    output_files = ["{}_MRDS_signal_fraction.nii.gz".format(args.out_prefix),
                    "{}_MRDS_evals.nii.gz".format(args.out_prefix),
                    "{}_MRDS_isotropic.nii.gz".format(args.out_prefix),
                    "{}_MRDS_num_tensors.nii.gz".format(args.out_prefix),
                    "{}_MRDS_evecs.nii.gz".format(args.out_prefix)]
    assert_outputs_exist(parser, args, output_files)
    assert_headers_compatible(parser, [args.in_volume] + [x for xs in mrds_files for x in xs])

    signal_fraction = []
    evals = []
    iso = []
    num_tensors = []
    evecs = []
    for N in range(3):
        signal_fraction.append(nib.load(mrds_files[N][0]).get_fdata(dtype=np.float32))
        evals.append(nib.load(mrds_files[N][1]).get_fdata(dtype=np.float32))
        iso.append(nib.load(mrds_files[N][2]).get_fdata(dtype=np.float32))
        num_tensors.append(nib.load(mrds_files[N][3]).get_fdata(dtype=np.float32))
        evecs.append(nib.load(mrds_files[N][4]).get_fdata(dtype=np.float32))

    # MOdel SElector MAP
    mosemap_img = nib.load(args.in_volume)
    mosemap = get_data_as_labels(mosemap_img)
    header = mosemap_img.header

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

    signal_fraction_out = np.zeros((X, Y, Z, 3))
    evals_out = np.zeros((X, Y, Z, 9))
    iso_out = np.zeros((X, Y, Z, 2))
    num_tensors_out = np.zeros((X, Y, Z), dtype=np.uint8)
    evecs_out = np.zeros((X, Y, Z, 9))

    # select data using mosemap
    for (X, Y, Z) in filtered_voxels:
        N = mosemap[X, Y, Z]-1

        # Maximum number of tensors is 3
        if N > 2:
            N = 2

        if N > -1:
            signal_fraction_out[X, Y, Z, :] = signal_fraction[N][X, Y, Z, :]
            evals_out[X, Y, Z, :] = evals[N][X, Y, Z, :]
            iso_out[X, Y, Z, :] = iso[N][X, Y, Z, :]
            num_tensors_out[X, Y, Z] = int(num_tensors[N][X, Y, Z])
            evecs_out[X, Y, Z, :] = evecs[N][X, Y, Z, :]

    # write output files
    nib.save(nib.Nifti1Image(signal_fraction_out,
                             affine=affine,
                             header=header,
                             dtype=np.float32), output_files[0])
    nib.save(nib.Nifti1Image(evals_out,
                             affine=affine,
                             header=header,
                             dtype=np.float32), output_files[1])
    nib.save(nib.Nifti1Image(iso_out,
                             affine=affine,
                             header=header,
                             dtype=np.float32), output_files[2])
    nib.save(nib.Nifti1Image(num_tensors_out,
                             affine=affine,
                             header=header,
                             dtype=np.uint8), output_files[3])
    nib.save(nib.Nifti1Image(evecs_out,
                             affine=affine,
                             header=header,
                             dtype=np.float32), output_files[4])


if __name__ == '__main__':
    main()
