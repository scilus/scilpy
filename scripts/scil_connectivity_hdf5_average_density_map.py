#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a density map for each connection from a hdf5 file.
Typically use after scil_tractogram_segment_bundles_for_connectivity.py in
order to obtain the average density map of each connection to allow the use
of --similarity in scil_connectivity_compute_matrices.py.

This script is parallelized, but will run much slower on non-SSD if too many
processes are used. The output is a directory containing the thousands of
connections:
out_dir/
    |-- LABEL1_LABEL1.nii.gz
    |-- LABEL1_LABEL2.nii.gz
    |-- [...]
    |-- LABEL90_LABEL90.nii.gz

Formerly: scil_compute_hdf5_average_density_map.py
"""

import argparse
import itertools
import logging
import multiprocessing
import os

import h5py
import numpy as np
import nibabel as nib

from scilpy.io.hdf5 import assert_header_compatible_hdf5, \
    reconstruct_streamlines_from_hdf5
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             add_processes_arg, assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             validate_nbr_processes)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_hdf5', nargs='+',
                   help='List of HDF5 filenames (.h5) from '
                        'scil_tractogram_segment_bundles_for_connectivity.py.')
    p.add_argument('out_dir',
                   help='Path of the output directory.')

    p.add_argument('--binary', action='store_true',
                   help='Binarize density maps before the population average.')

    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def _average_wrapper(args):
    hdf5_filenames = args[0]
    key = args[1]
    binary = args[2]
    out_dir = args[3]

    with h5py.File(hdf5_filenames[0], 'r') as hdf5_file_ref:
        affine = hdf5_file_ref.attrs['affine']
        dimensions = hdf5_file_ref.attrs['dimensions']
        density_data = np.zeros(dimensions, dtype=np.float32)

    for hdf5_filename in hdf5_filenames:
        with h5py.File(hdf5_filename, 'r') as hdf5_file:
            assert_header_compatible_hdf5(hdf5_file, (affine, dimensions))

            # scil_tractogram_segment_bundles_for_connectivity.py saves the
            # streamlines in VOX/CORNER
            streamlines = reconstruct_streamlines_from_hdf5(hdf5_file[key])
            if len(streamlines) == 0:
                continue
            density = compute_tract_counts_map(streamlines, dimensions)

        if binary:
            density_data[density > 0] += 1
        elif np.max(density) > 0:
            density_data += density / np.max(density)

    if np.max(density_data) > 0:
        density_data /= len(hdf5_filenames)

        nib.save(nib.Nifti1Image(density_data, affine),
                 os.path.join(out_dir, '{}.nii.gz'.format(key)))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_hdf5)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)

    keys = []
    for filename in args.in_hdf5:
        with h5py.File(filename, 'r') as curr_file:
            keys.extend(curr_file.keys())

    keys = set(keys)
    nbr_cpu = validate_nbr_processes(parser, args)
    if nbr_cpu == 1:
        for key in keys:
            _average_wrapper([args.in_hdf5, key, args.binary, args.out_dir])
    else:
        pool = multiprocessing.Pool(nbr_cpu)
        _ = pool.map(_average_wrapper,
                     zip(itertools.repeat(args.in_hdf5),
                         keys,
                         itertools.repeat(args.binary),
                         itertools.repeat(args.out_dir)))
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
