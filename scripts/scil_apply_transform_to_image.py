#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Transform *.nii or *.nii.gz using an affine/rigid transformation.
    For more information on how to use the various registration scripts
    see the doc/tractogram_registration.md readme file
"""

import argparse

import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.image import transform_anatomy


def _build_args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_file',
                   help='Path of the file to be transformed (nii or nii.gz)')

    p.add_argument('ref_file',
                   help='Path of the reference file (the static \n'
                   'file from registration), must be in the Nifti format.')

    p.add_argument('transformation',
                   help='Path of the file containing the 4x4 \n'
                   'transformation, matrix (*.npy).')

    p.add_argument('out_name',
                   help='Output filename of the transformed data.')

    p.add_argument('--inverse', action='store_true',
                   help='Apply the inverse transformation.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_file, args.ref_file,
                                 args.transformation])
    assert_outputs_exist(parser, args, args.out_name)

    transfo = np.loadtxt(args.transformation)
    if args.inverse:
        transfo = np.linalg.inv(transfo)

    _, ref_extension = split_name_with_nii(args.ref_file)
    _, in_extension = split_name_with_nii(args.in_file)
    if ref_extension not in ['.nii', '.nii.gz']:
        parser.error('{} is an unsupported format.'.format(args.ref_file))
    if in_extension not in ['.nii', '.nii.gz']:
        parser.error('{} is an unsupported format.'.format(args.in_file))

    transform_anatomy(transfo, args.ref_file, args.in_file,
                      args.out_name)


if __name__ == "__main__":
    main()
