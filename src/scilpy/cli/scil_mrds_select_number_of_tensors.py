#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use the NUFO map information to select the plausible number of tensors
in the Multi-Resolution Discrete Search (MRDS).
https://link.springer.com/chapter/10.1007/978-3-031-47292-3_4

scil_mrds_select_number_of_tensors uses the output from mdtmrds command.
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
        scil_mrds_select_number_of_tensors sub-01 nufo.nii.gz
"""

import argparse
import itertools
import logging

import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.io.image import get_data_as_mask
from scilpy.io.stateful_image import StatefulImage
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

    # MOdel SElector MAP
    mosemap_simg = StatefulImage.load(args.in_volume)
    mosemap = get_data_as_labels(mosemap_simg)
    X, Y, Z = mosemap.shape[0:3]

    signal_fraction = []
    evals = []
    iso = []
    num_tensors = []
    evecs = []
    for N in range(3):
        sf_simg = StatefulImage.load(mrds_files[N][0])
        sf_simg.reorient(mosemap_simg.axcodes)
        signal_fraction.append(sf_simg.get_fdata(dtype=np.float32))

        evals_simg = StatefulImage.load(mrds_files[N][1])
        evals_simg.reorient(mosemap_simg.axcodes)
        evals.append(evals_simg.get_fdata(dtype=np.float32))

        iso_simg = StatefulImage.load(mrds_files[N][2])
        iso_simg.reorient(mosemap_simg.axcodes)
        iso.append(iso_simg.get_fdata(dtype=np.float32))

        num_tensors_simg = StatefulImage.load(mrds_files[N][3])
        num_tensors_simg.reorient(mosemap_simg.axcodes)
        num_tensors.append(num_tensors_simg.get_fdata(dtype=np.float32))

        # evecs are stored in world space. A plain grid reslice does not change their values.
        evecs_simg = StatefulImage.load(mrds_files[N][4])
        evecs_simg.reorient(mosemap_simg.axcodes)
        evecs.append(evecs_simg.get_fdata(dtype=np.float32))

    # load mask
    if args.mask:
        mask_simg = StatefulImage.load(args.mask)
        mask_simg.reorient(mosemap_simg.axcodes)
        mask = get_data_as_mask(mask_simg, dtype=bool)
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
    # evecs are in world space. No direction rotation is required on save.
    StatefulImage.create_from(
        signal_fraction_out.astype(np.float32), mosemap_simg,
        is_orientation=False).save(output_files[0])
    StatefulImage.create_from(
        evals_out.astype(np.float32), mosemap_simg,
        is_orientation=False).save(output_files[1])
    StatefulImage.create_from(
        iso_out.astype(np.float32), mosemap_simg,
        is_orientation=False).save(output_files[2])
    StatefulImage.create_from(
        num_tensors_out.astype(np.uint8), mosemap_simg,
        is_orientation=False).save(output_files[3])
    StatefulImage.create_from(
        evecs_out.astype(np.float32), mosemap_simg,
        is_orientation=False).save(output_files[4])


if __name__ == '__main__':
    main()
