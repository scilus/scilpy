#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converts the datatype of an image volume.
"""

import argparse
import logging
import os

from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.image.operations import (get_image_ops, get_operations_doc)
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_outputs_exist)
from scilpy.utils.util import is_float


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('in_image',
                   help='The input file')
    p.add_argument('out_image',
                   help='Output image path.')

    p.add_argument('data_type',
                   help='Data type of the output image. Use the format: '
                        'uint8, int16, int/float32, int/float64.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    img = nib.load(args.in_image)
    dtype = img.header.get_data_dtype()
    print('Converting from {} to {}.'.format(dtype, args.data_type))

    data = img.get_fdata().astype(args.data_type)
    img.header.set_data_dtype(args.data_type)

    new_img = nib.Nifti1Image(data, img.affine, header=img.header)
    nib.save(new_img, args.out_image)
