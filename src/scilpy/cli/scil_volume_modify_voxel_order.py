#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Change the voxel order (strides) of a NIfTI image.

This script allows you to change the voxel order of a NIfTI image by modifying
its header. The voxel order, also known as strides, defines the orientation of
the image data in memory. This can be useful for compatibility with different
software packages that expect a specific voxel order.
In contrast, `scil_volume_flip` only flips the data array in memory,
without changing the header's orientation information.

The new voxel order can be specified in several ways:
- As a string of 3 characters, e.g., 'RAS', 'LPS', 'ASR'.
- As a comma-separated string of 3 characters, e.g., 'R,A,S'.
- As a string of 3 numbers, e.g., '123', '231', '-12-3'.
- As a comma-separated string of 3 numbers, e.g., '1,2,3', '-1,2,-3'.

For numeric input, 1, 2, and 3 correspond to the R, A, and S axes of the
image when loaded in RAS orientation. A negative sign flips the axis.
For example., '-1,2,-3' would correspond to a voxel order of 'LAS'.

For 4D images, the voxel order must be specified numerically.
e.g., '1,2,3,4' or '1,2,3' (if the 4th dimension is time and does not
need to be reordered). The 4th dimension must be 4 or -4.

To change the header of a tractogram (.trk), we recommend converting it to a
.tck file, then converting it back to .trk with the target NIfTI image as a
reference.
"""

import argparse
import logging
import nibabel as nib

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.orientation import parse_voxel_order
from scilpy.io.stateful_image import StatefulImage
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_image',
                   help='Path of the NIfTI file to modify.')
    p.add_argument('out_image',
                   help='Path of the modified NIfTI file to write.')
    p.add_argument('--new_voxel_order', required=True,
                   help='The new voxel order (e.g., "RAS", "1,2,3").')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image)
    assert_outputs_exist(parser, args, args.out_image)

    img = nib.load(args.in_image)
    simg = StatefulImage.load(args.in_image, to_orientation='RAS')

    parsed_voxel_order = parse_voxel_order(args.new_voxel_order,
                                           dimensions=len(img.shape))

    simg.reorient(parsed_voxel_order)

    nib.save(simg, args.out_image)


if __name__ == "__main__":
    main()
