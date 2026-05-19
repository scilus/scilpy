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
import numpy as np

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

    p.add_argument('--in_bvec',
                   help='Path of the b-vectors file.')
    p.add_argument('--in_bval',
                   help='Path of the b-values file.')
    p.add_argument('--out_bvec',
                   help='Path of the modified b-vectors file to write.')
    p.add_argument('--out_bval',
                   help='Path of the modified b-values file to write.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image, args.in_bvec)
    assert_outputs_exist(parser, args, args.out_image, args.out_bvec)

    img = nib.load(args.in_image)
    simg = StatefulImage.load(args.in_image)

    if args.in_bvec:
        bvecs = np.loadtxt(args.in_bvec)
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T

        # Create dummy bvals to satisfy StatefulImage validation
        bvals = np.zeros(len(bvecs))
        simg.attach_gradients(bvals, bvecs)

    parsed_voxel_order = parse_voxel_order(args.new_voxel_order,
                                           dimensions=len(img.shape))

    simg.reorient(parsed_voxel_order)

    # To enforce the new voxel order in the header, we need to create
    # a new StatefulImage, which will update the header accordingly.
    new_simg = StatefulImage.convert_to_simg(simg, simg.bvals, simg.bvecs)
    new_simg.save(args.out_image)

    if args.in_bvec and args.out_bvec:
        if args.in_bval and args.out_bval:
            new_simg.save_gradients(args.out_bval, args.out_bvec)
        else:
            # If no bval file or no output bval path, save only bvecs.
            # new_simg.bvecs returns bvecs in the current (new) orientation.
            np.savetxt(args.out_bvec, new_simg.bvecs.T, fmt='%.8f')
            if args.in_bval and not args.out_bval:
                logging.warning("b-values were provided but no output path "
                                "was specified. b-values will not be saved.")


if __name__ == "__main__":
    main()
