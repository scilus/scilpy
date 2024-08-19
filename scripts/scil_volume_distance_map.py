#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute distance map between two binary masks using KDTree. The distance map
is the Euclidean distance from each voxel of the first mask to the closest
voxel of the second mask.
It's worst case scenario is two very big masks that are far appart or a very
small mask that is completely inside a very big mask.

Take this command as an example:
  scil_volume_distance_map.py AF_L.nii.gz brain_mask.nii.gz \
    AF_L_to_brain_mask.nii.gz

We have a bundle and brain mask, the first is 100% inside the second.
The output will be a distance map from the bundle to the brain mask, which
will be all zeros.

If we take the brain mask as the first input and the bundle as the second,
the output will be a distance map from the brain mask to the bundle.

If you want both distance maps at once, you can use the --symmetric_distance
option.
"""

import argparse
import logging

import nibabel as nib
import numpy as np
from scipy.spatial import KDTree

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import add_overwrite_arg, add_verbose_arg, \
    assert_headers_compatible, assert_inputs_exist, assert_outputs_exist


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_mask_1', metavar='IN_SOURCE',
                   help='Input file name, in nifti format.')
    p.add_argument('in_mask_2', metavar='IN_TARGET',
                   help='Input file name, in nifti format.')
    p.add_argument('out_distance', metavar='OUT_DISTANCE_MAP',
                   help='Input file name, in nifti format.')

    p.add_argument('--symmetric_distance', action='store_true',
                   help='Compute the distance from mask 1 to mask 2 and the '
                        'distance from mask 2 to mask 1 and sum them up.')
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_mask_1, args.in_mask_2])
    assert_outputs_exist(parser, args, args.out_distance)
    assert_headers_compatible(parser, [args.in_mask_1, args.in_mask_2])

    img_1 = nib.load(args.in_mask_1)
    img_2 = nib.load(args.in_mask_2)

    mask_1 = get_data_as_mask(img_1)
    mask_2 = get_data_as_mask(img_2)
    logging.debug(f'Loaded two masks with {np.count_nonzero(mask_1)} and '
                  f'{np.count_nonzero(mask_2)} voxels')

    # Compute distance map using KDTree
    tree = KDTree(np.argwhere(mask_2))
    distance_map = np.zeros(mask_1.shape)
    distance = tree.query(np.argwhere(mask_1))[0]
    distance_map[np.where(mask_1)] = distance

    if args.symmetric_distance:
        # Compute the symmetric distance map and merge it with the previous one
        tree = KDTree(np.argwhere(mask_1))
        distance = tree.query(np.argwhere(mask_2))[0]
        distance_map[np.where(mask_2)] = distance

    out_img = nib.Nifti1Image(distance_map.astype(float), img_1.affine)
    nib.save(out_img, args.out_distance)


if __name__ == "__main__":
    main()
