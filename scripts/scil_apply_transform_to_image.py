#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform Nifti (.nii.gz) using an affine/rigid transformation.

For more information on how to use the registration script, follow this link:
https://scilpy.readthedocs.io/en/latest/documentation/tractogram_registration.html
"""

import argparse

import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, load_matrix_in_any_format)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.image import transform_anatomy


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_file',
                   help='Path of the file to be transformed (nii or nii.gz)')
    p.add_argument('in_target_file',
                   help='Path of the reference target file (.nii.gz).')
    p.add_argument('in_transfo',
                   help='Path of the file containing the 4x4 \n'
                        'transformation, matrix (.txt, .npy or .mat).')
    p.add_argument('out_name',
                   help='Output filename of the transformed data.')

    p.add_argument('--inverse', action='store_true',
                   help='Apply the inverse transformation.')

    p.add_argument('--keep_dtype', action='store_true',
                   help='If True, keeps the data_type of the input image '
                        '(in_file) when saving the output image (out_name).')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_file, args.in_target_file,
                                 args.in_transfo])
    assert_outputs_exist(parser, args, args.out_name)

    transfo = load_matrix_in_any_format(args.in_transfo)
    if args.inverse:
        transfo = np.linalg.inv(transfo)

    _, ref_extension = split_name_with_nii(args.in_target_file)
    _, in_extension = split_name_with_nii(args.in_file)
    if ref_extension not in ['.nii', '.nii.gz']:
        parser.error('{} is an unsupported format.'.format(args.in_target_file))
    if in_extension not in ['.nii', '.nii.gz']:
        parser.error('{} is an unsupported format.'.format(args.in_file))

    transform_anatomy(transfo, args.in_target_file, args.in_file,
                      args.out_name, keep_dtype=args.keep_dtype)


if __name__ == "__main__":
    main()
