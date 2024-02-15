#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reshape / reslice / resample *.nii or *.nii.gz using a reference.
This script can be used to align freesurfer/civet output, as .mgz,
to the original input image.

>>> scil_volume_reshape_to_reference.py wmparc.mgz t1.nii.gz wmparc_t1.nii.gz\\
    --interpolation nearest

Formerly: scil_reshape_to_reference.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.image.volume_operations import apply_transform
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             add_verbose_arg, assert_outputs_exist)


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

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_file, args.in_ref_file])
    assert_outputs_exist(parser, args, args.out_file)

    # Load images.
    in_file = nib.load(args.in_file)
    ref_file = nib.load(args.in_ref_file)

    reshaped_img = apply_transform(np.eye(4), ref_file, in_file,
                                   interp=args.interpolation,
                                   keep_dtype=args.keep_dtype)

    nib.save(reshaped_img, args.out_file)


if __name__ == "__main__":
    main()
