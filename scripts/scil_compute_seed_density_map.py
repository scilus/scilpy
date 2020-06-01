#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a density map of seeds saved in .trk file.
"""

import argparse

from dipy.io.streamline import load_tractogram
from nibabel import Nifti1Image
from nibabel.streamlines import detect_format, TrkFile
import numpy as np
from scilpy.io.utils import (
    add_overwrite_arg,
    assert_inputs_exist,
    assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tractogram_filename',
                   help='Tracts filename. Format must be .trk.')
    p.add_argument('seed_density_filename',
                   help='Output seed density filename. Format must be Nifti.')
    p.add_argument('--binary',
                   metavar='FIXED_VALUE', type=int, nargs='?', const=1,
                   help='If set, will store the same value for all '
                        'intersected voxels, creating a binary map.\nWhen set '
                        'without a value, 1 is used.\n If a value is given, '
                        'will be used as the stored value.')
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.tractogram_filename])
    assert_outputs_exist(parser, args, [args.seed_density_filename])

    tracts_format = detect_format(args.tractogram_filename)
    if tracts_format is not TrkFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk): {0}".format(args.tractogram_filename))

    max_ = np.iinfo(np.int16).max
    if args.binary is not None and (args.binary <= 0 or args.binary > max_):
        parser.error('The value of --binary ({}) '
                     'must be greater than 0 and smaller or equal to {}'
                     .format(args.binary, max_))

    # Load files and data. TRKs can have 'same' as reference
    # Can handle streamlines outside of bbox
    sft = load_tractogram(args.tractogram_filename, 'same',
                          bbox_valid_check=False)
    # Streamlines are saved in RASMM but seeds are saved in VOX
    # This might produce weird behavior with non-iso
    sft.to_vox()
    sft.to_corner()
    if 'seeds' in sft.data_per_streamline:
        seeds = sft.data_per_streamline['seeds']
    else:
        parser.error('Tractogram does not contain seeds')

    # Create seed density map
    _, shape, _, _ = sft.space_attributes
    seed_density = np.zeros(shape, dtype=np.int32)
    for seed in seeds:
        # Set value at mask, either binary or increment
        seed_voxel = np.round(seed).astype(np.int)
        if args.binary is not None:
            seed_density[tuple(seed_voxel)] = args.binary
        else:
            seed_density[tuple(seed_voxel)] += 1

    # Save seed density map
    dm_img = Nifti1Image(seed_density.astype(np.int32),
                         sft.affine)
    dm_img.to_filename(args.seed_density_filename)


if __name__ == '__main__':
    main()
