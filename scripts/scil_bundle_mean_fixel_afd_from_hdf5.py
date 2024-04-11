#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the mean Apparent Fiber Density (AFD) and mean Radial fODF (radfODF)
maps for every connections within a hdf5 (.h5) file.

This is the "real" fixel-based fODF amplitude along every streamline
of each connection, averaged at every voxel.

Please use a hdf5 (.h5) file containing decomposed connections

Formerly: scil_compute_fixel_afd_from_hdf5.py
"""

import argparse
import itertools
import logging
import multiprocessing
import os
import shutil

import h5py
import nibabel as nib
import numpy as np

from scilpy.io.hdf5 import (assert_header_compatible_hdf5,
                            reconstruct_sft_from_hdf5)
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_sh_basis_args, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             parse_sh_basis_arg, validate_nbr_processes)
from scilpy.tractanalysis.afd_along_streamlines \
    import afd_map_along_streamlines

EPILOG = """
Reference:
    [1] Raffelt, D., Tournier, JD., Rose, S., Ridgway, GR., Henderson, R.,
        Crozier, S., Salvado, O., & Connelly, A. (2012).
        Apparent Fibre Density: a novel measure for the analysis of
        diffusion-weighted magnetic resonance images. NeuroImage,
        59(4), 3976--3994.
"""


def _afd_rd_wrapper(args):
    in_hdf5_filename = args[0]
    key = args[1]
    fodf_img = args[2]
    sh_basis = args[3]
    length_weighting = args[4]
    is_legacy = args[5]

    with h5py.File(in_hdf5_filename, 'r') as in_hdf5_file:
        sft, _ = reconstruct_sft_from_hdf5(in_hdf5_file, key)

    if sft is None:  # No streamlines in group
        return key, 0

    afd_mean_map, rd_mean_map = afd_map_along_streamlines(sft, fodf_img,
                                                          sh_basis,
                                                          length_weighting,
                                                          is_legacy=is_legacy)
    afd_mean = np.average(afd_mean_map[afd_mean_map > 0])

    return key, afd_mean


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_hdf5',
                   help='HDF5 filename (.h5) containing decomposed '
                        'connections.')
    p.add_argument('in_fodf',
                   help='Path of the fODF volume in spherical harmonics (SH).')
    p.add_argument('out_hdf5',
                   help='Path of the output HDF5 filenames (.h5).')

    p.add_argument('--length_weighting', action='store_true',
                   help='If set, will weigh the AFD values according to '
                        'segment lengths. [%(default)s]')

    add_processes_arg(p)
    add_sh_basis_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_hdf5, args.in_fodf])
    assert_outputs_exist(parser, args, [args.out_hdf5])

    nbr_cpu = validate_nbr_processes(parser, args)
    sh_basis, is_legacy = parse_sh_basis_arg(args)

    # HDF5 will not overwrite the file
    if os.path.isfile(args.out_hdf5):
        os.remove(args.out_hdf5)

    fodf_img = nib.load(args.in_fodf)
    with h5py.File(args.in_hdf5, 'r') as in_hdf5_file:
        assert_header_compatible_hdf5(in_hdf5_file, fodf_img)
        keys = list(in_hdf5_file.keys())
        in_hdf5_file.close()

    if nbr_cpu == 1:
        results_list = []
        for key in keys:
            results_list.append(_afd_rd_wrapper([args.in_hdf5, key, fodf_img,
                                                 sh_basis,
                                                 args.length_weighting,
                                                 is_legacy]))

    else:
        pool = multiprocessing.Pool(nbr_cpu)
        results_list = pool.map(_afd_rd_wrapper,
                                zip(itertools.repeat(args.in_hdf5),
                                    keys,
                                    itertools.repeat(fodf_img),
                                    itertools.repeat(sh_basis),
                                    itertools.repeat(args.length_weighting),
                                    itertools.repeat(is_legacy)))
        pool.close()
        pool.join()

    shutil.copy(args.in_hdf5, args.out_hdf5)
    with h5py.File(args.out_hdf5, 'a') as out_hdf5_file:
        for key, afd_fixel in results_list:
            group = out_hdf5_file[key]
            if 'afd_fixel' in group:
                del group['afd_fixel']
            group.create_dataset('afd_fixel', data=afd_fixel)


if __name__ == '__main__':
    main()
