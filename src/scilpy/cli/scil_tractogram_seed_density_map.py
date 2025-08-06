#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a density map of seeds saved in .trk file.

* They are saved as a data_per_streamline 'seeds' when using our tractography
scripts with option --save_seeds.
- scil_tracking_local.py
- scil_tracking_local_dev.py
- scil_tracking_pft.py.

If you use your own tractograms, you can create a 'seeds' data_per_streamline.
Don't forget to save the coordinates in voxel space, corner origin.

Formerly: scil_compute_seed_density_map.py
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
                             assert_outputs_exist, ranged_type)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('tractogram_filename',
                   help='Tracts filename. Format must be .trk. \nFile should '
                        'contain a "seeds" value in the data_per_streamline.\n'
                        'These seeds must be in space: voxel, origin: corner.')
    p.add_argument('seed_density_filename',
                   help='Output seed density filename. Format must be Nifti.')
    p.add_argument('--binary',
                   metavar='FIXED_VALUE',
                   type=ranged_type(int, 0, np.iinfo(np.int16).max,
                                    min_excluded=True),
                   nargs='?', const=1,
                   help='If set, will store the same value for all intersected'
                        ' voxels, creating a binary map.\n'
                        'When set without a value, 1 is used (and dtype '
                        'uint8).\nIf a value is given, it will be used as the '
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
    assert_outputs_exist(parser, args, [args.seed_density_filename])

    tracts_format = detect_format(args.tractogram_filename)
    if tracts_format is not TrkFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk): {0}".format(args.tractogram_filename))

    # Load files and data. TRKs can have 'same' as reference
    # Can handle streamlines outside of bbox, if asked by user.
    logging.info("Loading tractogram")
    sft = load_tractogram(args.tractogram_filename, 'same',
                          bbox_valid_check=args.bbox_check)

    # IMPORTANT
    # Origin should be center when creating the seeds (see below, we
    # are using round, not floor; works with center origin), which is the
    # default in dipy. Can NOT be verified here.

    # P.s. Streamlines are saved in RASMM by nibabel by default but nibabel
    # does not change the values in data_per_streamline. They are also not
    # impacted by methods in the sft such as to_vox or to_corner.
    # data_per_streamline is thus always exactly as it was when created by
    # user.

    # SFT shape and affine are fixed, no matter the sft space and origin.
    affine, shape, _, _ = sft.space_attributes

    # Get the seeds
    if 'seeds' not in sft.data_per_streamline:
        parser.error('Tractogram does not contain seeds.')

    # Create seed density map
    seed_density = np.zeros(shape, dtype=np.int32)
    for seed in sft.data_per_streamline['seeds']:
        # Set value at mask, either binary or increment
        seed_voxel = np.round(seed).astype(int)
        dtype_to_use = np.int32
        if args.binary is not None:
            if args.binary == 1:
                dtype_to_use = np.uint8
            seed_density[tuple(seed_voxel)] = args.binary
        else:
            seed_density[tuple(seed_voxel)] += 1

    # Save seed density map
    logging.info("Saving out density map.")
    dm_img = Nifti1Image(seed_density.astype(dtype_to_use), affine)
    dm_img.to_filename(args.seed_density_filename)


if __name__ == '__main__':
    main()
