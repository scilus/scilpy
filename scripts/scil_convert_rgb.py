#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converts a RGB image encoded as a 4D image to a RGB image encoded as
a 3D image.

Typically, most software tools used in the SCIL (including MI-Brain) use
the former, while Trackvis uses the latter.
"""

import argparse

from dipy.io.utils import decfa, decfa_to_float
import nibabel as nb

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('in_image',
                   help='name of input 4D RGB image.\n' +
                        'This is a 4D image where the 4th dimension contains '
                        '3 values.')
    p.add_argument('out_image',
                   help='name of output 3D RGB image, in Trackvis format.\n'
                        'This is a 3D image where each voxel contains a ' +
                        'tuple of 3 elements, one for each value.')
    add_overwrite_arg(p)

    return p


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_image])
    assert_outputs_exist(parser, args, [args.out_image])

    original_im = nb.load(args.in_image)

    if len(original_im.get_shape()) == 4:
        if "float" in original_im.header.get_data_shape():
            scale = True
        else:
            scale = False

        converted_im = decfa(original_im, scale=scale)

        nb.save(converted_im, args.out_image)

    elif len(original_im.get_shape()) == 3:
        converted_im_float = decfa_to_float(original_im)

        converted_data = converted_im_float.get_fdata()
        converted_data_int = converted_data.astype("uint8")

        converted_im = nb.Nifti1Image(converted_data_int,
                                      converted_im_float.affine)

        nb.save(converted_im, args.out_image)


if __name__ == "__main__":
    main()
