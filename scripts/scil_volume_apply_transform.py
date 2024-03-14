#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform Nifti (.nii.gz) using an affine/rigid transformation.

For more information on how to use the registration script, follow this link:
https://scilpy.readthedocs.io/en/latest/documentation/tractogram_registration.html

Formerly: scil_apply_transform_to_image.py.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.image.volume_operations import apply_transform
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg,
                             load_matrix_in_any_format)
from scilpy.utils.filenames import split_name_with_nii


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

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, [args.in_file, args.in_target_file,
                                 args.in_transfo])
    assert_outputs_exist(parser, args, args.out_name)

    _, ref_extension = split_name_with_nii(args.in_target_file)
    _, in_extension = split_name_with_nii(args.in_file)
    if ref_extension not in ['.nii', '.nii.gz']:
        parser.error('{} is an unsupported format.'.format(
            args.in_target_file))
    if in_extension not in ['.nii', '.nii.gz']:
        parser.error('{} is an unsupported format.'.format(args.in_file))

    # Loading
    transfo = load_matrix_in_any_format(args.in_transfo)
    if args.inverse:
        transfo = np.linalg.inv(transfo)
    moving = nib.load(args.in_file)
    reference = nib.load(args.in_target_file)

    # Processing, saving
    warped_img = apply_transform(
        transfo, reference, moving, keep_dtype=args.keep_dtype)

    nib.save(warped_img, args.out_name)


if __name__ == "__main__":
    main()
