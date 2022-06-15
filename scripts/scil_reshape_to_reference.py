#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reshape / reslice / resample *.nii or *.nii.gz using a reference.
This script can be used to align freesurfer/civet output, as .mgz,
to the original input image.


>>> scil_reshape_to_reference.py wmparc.mgz t1.nii.gz wmparc_t1.nii.gz \\
    --interpolation nearest
"""

import argparse

import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.image import transform_anatomy


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_file',
                   help='Path of the image (.nii or .mgz) to be reshaped.')
    p.add_argument('in_ref_file',
                   help='Path of the reference image (.nii).')
    p.add_argument('out_file',
                   help='Output filename of the reshaped image (.nii).')

    p.add_argument('--interpolation', default='linear',
                   choices=['linear', 'nearest'],
                   help='Interpolation: "linear" or "nearest". [%(default)s]')

    p.add_argument('--keep_dtype', action='store_true',
                   help='If True, keeps the data_type of the input image '
                        '(in_file) when saving the output image (out_file).')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_file, args.in_ref_file])
    assert_outputs_exist(parser, args, args.out_file)

    transform_anatomy(np.eye(4), args.in_ref_file, args.in_file,
                      args.out_file, interp=args.interpolation,
                      keep_dtype=args.keep_dtype)


if __name__ == "__main__":
    main()
