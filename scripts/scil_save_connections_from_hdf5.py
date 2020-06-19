#!/usr/bin/env python3
# encoding: utf-8

"""
Save individual connection of an hd5f from scil_decompose_connectivity.py.
Useful for quality control and visual inspections.

The output is a directory containing the thousands of connections:
out_dir/
    ├── LABEL1_LABEL1.trk
    ├── LABEL1_LABEL2.trk
    ├── [...]
    └── LABEL90_LABEL90.trk
"""

import argparse
import os

from dipy.io.stateful_tractogram import Space, Origin, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import create_nifti_header
import h5py

from scilpy.io.streamlines import reconstruct_streamlines_from_hdf5
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_hdf5',
                   help='HDF5 filename (.h5) containing decomposed '
                        'connections.')
    p.add_argument('out_dir',
                   help='Path of the output directory.')

    p.add_argument('--include_dps', action='store_true',
                   help='Include the data_per_streamline the metadata.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_hdf5)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)

    hdf5_file = h5py.File(args.in_hdf5, 'r')
    for key in hdf5_file.keys():
        affine = hdf5_file.attrs['affine']
        dimensions = hdf5_file.attrs['dimensions']
        voxel_sizes = hdf5_file.attrs['voxel_sizes']
        streamlines = reconstruct_streamlines_from_hdf5(hdf5_file, key)
        header = create_nifti_header(affine, dimensions, voxel_sizes)
        sft = StatefulTractogram(streamlines, header, Space.VOX,
                                 origin=Origin.TRACKVIS)
        if args.include_dps:
            for dps_key in hdf5_file[key].keys():
                if dps_key not in ['data', 'offsets', 'lengths']:
                    sft.data_per_streamline[dps_key] = hdf5_file[key][dps_key]

        save_tractogram(sft, '{}.trk'.format(os.path.join(args.out_dir, key)))

    hdf5_file.close()


if __name__ == "__main__":
    main()
