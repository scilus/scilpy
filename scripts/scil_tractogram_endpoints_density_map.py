#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a density map of the endpoints.

See also scil_bundle_compute_endpoints_map.py: if the tractogram is an
uniformized bundle, produces a separate map for the head and the tail.

See also:
     scil_tractogram_seed_density_map.py
     scil_tractogram_compute_density_map.py
"""

import argparse
import logging

from dipy.io.streamline import load_tractogram
from nibabel import Nifti1Image
from nibabel.streamlines import detect_format, TrkFile
import numpy as np

from scilpy.io.utils import (add_bbox_arg,
                             add_verbose_arg,
                             add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tractogram_filename',
                   help='Tractogram.')
    p.add_argument('density_filename',
                   help='Output endpoints density filename. Format must be '
                        'Nifti.')
    p.add_argument('--binary',
                   metavar='FIXED_VALUE', type=int, nargs='?', const=1,
                   help='If set, will store the same value for all intersected'
                        ' voxels, creating a binary map.\n'
                        'When set without a value, 1 is used (and dtype '
                        'uint8).\nIf a value is given, will be used as the '
                        'stored value.')

    add_bbox_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.tractogram_filename])
    assert_outputs_exist(parser, args, [args.density_filename])

    max_ = np.iinfo(np.int16).max
    if args.binary is not None and (args.binary <= 0 or args.binary > max_):
        parser.error('The value of --binary ({}) '
                     'must be greater than 0 and smaller or equal to {}'
                     .format(args.binary, max_))

    # Load files and data. TRKs can have 'same' as reference
    # Can handle streamlines outside of bbox, if asked by user.
    logging.info("Loading tractogram.")
    sft = load_tractogram(args.tractogram_filename, 'same',
                          bbox_valid_check=args.bbox_check)
    affine, shape, _, _ = sft.space_attributes

    # Process
    sft.to_vox()
    sft.to_corner()  # With corner, using floor gives the voxel
    endpoints_density = np.zeros(shape, dtype=np.int32)
    for s in sft.streamlines:
        for p in [0, -1]:
            # Set value at mask, either binary or increment
            endpoint_voxel = np.floor(s[p, :]).astype(int)
            dtype_to_use = np.int32
            if args.binary is not None:
                if args.binary == 1:
                    dtype_to_use = np.uint8
                endpoints_density[tuple(endpoint_voxel)] = args.binary
            else:
                endpoints_density[tuple(endpoint_voxel)] += 1

    # Save density map
    logging.info("Saving density map: {}".format(args.density_filename))
    dm_img = Nifti1Image(endpoints_density.astype(dtype_to_use), affine)
    dm_img.to_filename(args.density_filename)


if __name__ == '__main__':
    main()
