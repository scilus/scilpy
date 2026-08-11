#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a linear affine transform into translation, rotation, scale and shear.

Accepted inputs include raw text 4x4 matrices, NumPy matrices, ANTs `.mat`
affines and ITK affine text transforms.

By default, each requested component is written as a homogeneous 4x4 matrix.
The decomposition follows:

    affine = translation @ rotation @ shear @ scale

Rotation can instead be saved as intrinsic Euler angles in `XYZ` order or as
Rodrigues angle-axis values. Angle-based outputs are written in radians.
"""

import argparse
import logging

import numpy as np
from scipy.spatial.transform import Rotation

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             load_matrix_in_any_format)
from scilpy.utils.spatial import split_affine_transform
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_transfo',
                   help='Path to the input affine transform.')
    p.add_argument('--translation',
                   help='Output text file for the translation component.')
    p.add_argument('--rotation',
                   help='Output text file for the rotation component.')
    p.add_argument('--scale',
                   help='Output text file for the scale component.')
    p.add_argument('--shear',
                   help='Output text file for the shear component.')

    output_mode = p.add_mutually_exclusive_group()
    output_mode.add_argument('--angles', action='store_true',
                             help='Save rotation as 3 intrinsic Euler angles '
                                  'in XYZ order (radians). Requires '
                                  '--rotation.')
    output_mode.add_argument('--rodrigues', action='store_true',
                             help='Save rotation as 4 values: angle followed '
                                  'by the 3 rotation-axis coordinates '
                                  '(radians). Requires --rotation.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _rotation_to_rodrigues(rotation_matrix, eps=1e-12):
    rotvec = Rotation.from_matrix(rotation_matrix[:3, :3]).as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < eps:
        axis = np.array([1., 0., 0.])
        angle = 0.
    else:
        axis = rotvec / angle
    return np.concatenate(([angle], axis))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    outputs = [args.translation, args.rotation, args.scale, args.shear]
    if np.all([out is None for out in outputs]):
        parser.error('No output selected. Choose at least one output option.')

    if (args.angles or args.rodrigues) and args.rotation is None:
        parser.error('Options --angles and --rodrigues require --rotation.')

    assert_inputs_exist(parser, args.in_transfo)
    assert_outputs_exist(parser, args, [], [out for out in outputs
                                            if out is not None])

    affine = load_matrix_in_any_format(args.in_transfo)
    translation, rotation, shear, scale = split_affine_transform(affine)

    if args.translation:
        np.savetxt(args.translation, translation, fmt='%.18e')
    if args.rotation:
        if args.angles:
            rotation_values = Rotation.from_matrix(
                rotation[:3, :3]).as_euler('XYZ', degrees=False)
        elif args.rodrigues:
            rotation_values = _rotation_to_rodrigues(rotation)
        else:
            rotation_values = rotation
        np.savetxt(args.rotation, rotation_values, fmt='%.18e')
    if args.scale:
        np.savetxt(args.scale, scale, fmt='%.18e')
    if args.shear:
        np.savetxt(args.shear, shear, fmt='%.18e')


if __name__ == "__main__":
    main()
