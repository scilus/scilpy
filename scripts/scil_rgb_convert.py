#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converts a RGB image encoded as a 4D image to a RGB image encoded as
a 3D image, or vice versa.

Typically, most software tools used in the SCIL (including MI-Brain) use
the former, while Trackvis uses the latter.

Input
-Case 1: 4D image where the 4th dimension contains 3 values.
-Case 2: 3D image, in Trackvis format where each voxel contains a
         tuple of 3 elements, one for each value.

Output
-Case 1: 3D image, in Trackvis format where each voxel contains a
         tuple of 3 elements, one for each value (uint8).
-Case 2: 4D image where the 4th dimension contains 3 values (uint8).

Formerly: scil_convert_rgb.py
"""

import argparse
import logging

from dipy.io.utils import decfa, decfa_to_float
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('in_image',
                   help='name of input RGB image.\n' +
                        'Either 4D or 3D image.')
    p.add_argument('out_image',
                   help='name of output RGB image.\n' +
                        'Either 3D or 4D image.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image)
    assert_outputs_exist(parser, args, args.out_image)

    original_im = nib.load(args.in_image)

    if original_im.ndim == 4:
        if "float" in original_im.header.get_data_shape():
            scale = True
        else:
            scale = False

        converted_im = decfa(original_im, scale=scale)

        nib.save(converted_im, args.out_image)

    elif original_im.ndim == 3:
        converted_im_float = decfa_to_float(original_im)

        converted_data_int = \
            np.asanyarray(converted_im_float.dataobj).astype(np.uint8)
        converted_im = nib.Nifti1Image(converted_data_int,
                                       converted_im_float.affine)

        nib.save(converted_im, args.out_image)


if __name__ == "__main__":
    main()
