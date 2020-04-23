#!/usr/bin/env python
# encoding: utf-8

"""
Compute a density map from a streamlines file.

A specific value can be assigned instead of using the tract count.

This script correctly handles compressed streamlines.
"""
import argparse
import os

from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
import h5py
import numpy as np
import nibabel as nib

from scilpy.io.streamlines import reconstruct_streamlines_from_hdf5
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_hdf5', nargs='+',
                   help='List of HDF5 filenames (.h5) containing streamlines '
                        'data, offsets and lengths.')
    p.add_argument('population_template',
                        help='Reference anatomy for the streamlines '
                             '(.nii or .nii.gz).')
    p.add_argument('out_dir',
                   help='Path of the output directory.')

    p.add_argument('--binary', action='store_true',
                   help='Binarize density maps before the population average.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_hdf5+[args.population_template])
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)

    hdf5_files = []
    keys = []
    for filename in args.in_hdf5:
        curr_file = h5py.File(filename, 'r')
        hdf5_files.append(curr_file)
        keys.extend(curr_file.keys())


    template_img = nib.load(args.population_template)
    dimensions = template_img.shape
    for key in set(keys):
        density_data = np.zeros(dimensions)
        for hdf5_file in hdf5_files:
            # scil_decompose_connectivity.py saves the streamlines in VOX/CORNER
            streamlines = reconstruct_streamlines_from_hdf5(hdf5_file, key)
            density = compute_tract_counts_map(streamlines, dimensions)

            if args.binary:
                density_data[density > 0] += 1
            else:
                density_data += density / np.max(density)
        density_data /= np.max(density_data)
        nib.save(nib.Nifti1Image(density_data, template_img.affine),
                 os.path.join(args.out_dir, '{}.nii.gz'.format(key)))


if __name__ == "__main__":
    main()
