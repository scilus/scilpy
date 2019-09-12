#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute a density map of seeds saved in .trk file.
"""

import argparse

from nibabel import Nifti1Image
from nibabel.affines import apply_affine
from nibabel.streamlines import (
    detect_format,
    Field,
    load,
    TckFile)
import numpy as np
from scilpy.io.utils import (
    add_overwrite_arg,
    assert_inputs_exist,
    assert_outputs_exist)


def _build_args_parser():
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
    p.add_argument('--lazy_load', action='store_true',
                   help='Load the file in lazy-loading')
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.tractogram_filename])
    assert_outputs_exist(parser, args, [args.seed_density_filename])

    tracts_format = detect_format(args.tractogram_filename)
    if tracts_format is TckFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk): {0}".format(args.tractogram_filename))

    max_ = np.iinfo(np.int16).max
    if args.binary is not None and (args.binary <= 0 or args.binary > max_):
        parser.error('The value of --binary ({}) '
                     'must be greater than 0 and smaller or equal to {}'
                     .format(args.binary, max_))

    # Load tractogram and load seeds
    tracts_file = load(args.tractogram_filename, args.lazy_load)
    if 'seeds' in tracts_file.tractogram.data_per_streamline:
        seeds = tracts_file.tractogram.data_per_streamline['seeds']
    else:
        parser.error('Tractogram does not contain seeds')

    # Transform seeds if they're all in memory
    if not args.lazy_load:
        seeds = apply_affine(np.linalg.inv(tracts_file.affine), seeds)

    # Create seed density map
    shape = tracts_file.header[Field.DIMENSIONS]
    seed_density = np.zeros(shape, dtype=np.int32)
    for seed in seeds:
        # If seeds are lazily loaded, we have to transform them
        # as they get loaded
        if args.lazy_load:
            seed = apply_affine(np.linalg.inv(tracts_file.affine), seed)

        # Set value at mask, either binary or increment
        seed_voxel = np.round(seed).astype(np.int)
        if args.binary is not None:
            seed_density[tuple(seed_voxel)] = args.binary
        else:
            seed_density[tuple(seed_voxel)] += 1

    # Save seed density map
    dm_img = Nifti1Image(seed_density.astype(np.int32),
                         tracts_file.affine)
    dm_img.to_filename(args.seed_density_filename)


if __name__ == '__main__':
    main()
