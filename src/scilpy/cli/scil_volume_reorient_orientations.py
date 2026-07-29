#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reorient directional data (SH or peaks) in a volume between world space and voxel space.
This script DOES NOT change the voxel grid order, only the spatial orientation
of what is written inside the voxels.

Legacy Dipy format used to store orientations relative to the voxel grid (voxel space).
MRtrix and modern Scilpy store orientations relative to the scanner world coordinates
(world space), making them invariant to the voxel grid's ordering.

Use this script to update legacy data to world space, or convert back to voxel space if needed.
"""

import argparse
import logging
import sys

import nibabel as nib
import numpy as np

from scilpy.io.stateful_image import StatefulImage
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             add_sh_basis_args, assert_inputs_exist,
                             assert_outputs_exist, parse_sh_basis_arg)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_image',
                   help='Input image containing directional data (.nii.gz).')
    p.add_argument('out_image',
                   help='Output image (.nii.gz).')

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--to_world', action='store_true',
                       help='Rotate orientations from voxel space to world space.')
    group.add_argument('--to_voxel', action='store_true',
                       help='Rotate orientations from world space to voxel space.')

    add_sh_basis_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image)
    assert_outputs_exist(parser, args, args.out_image)

    sh_basis, is_legacy = parse_sh_basis_arg(args)

    simg = StatefulImage.load(args.in_image, to_orientation=None,
                              is_orientation=True,
                              sh_basis=sh_basis,
                              is_legacy=is_legacy)

    if args.to_world:
        logging.info("Rotating directional data to world space.")
        rotated_data = simg.to_world_direction()
    else:
        logging.info("Rotating directional data to voxel space.")
        rotated_data = simg.to_voxel_direction()

    # Save as a standard Nifti1Image with the same affine and header
    out_img = nib.Nifti1Image(rotated_data, simg.affine, simg.header)
    nib.save(out_img, args.out_image)


if __name__ == '__main__':
    main()
