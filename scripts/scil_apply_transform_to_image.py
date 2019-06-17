#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.image import transform_anatomy


DESCRIPTION = """
    Transform *.nii or *.nii.gz using an affine/rigid transformation.

    For more informations on how to use the various registration scripts
    see the doc/tractogram_registration.md readme file
"""


def _buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)

    p.add_argument('in_file', action='store', metavar='IN_FILE',
                   type=str, help='Path of the file that will be transformed')

    p.add_argument('ref_file', action='store', metavar='REF_FILE',
                   type=str, help='Path of the reference file (the static \n'
                                  'file from registration), \n'
                                  'must be in the Nifti format')

    p.add_argument('transformation', action='store', metavar='TRANSFORMATION',
                   type=str, help='Path of the file containing the 4x4 \n'
                                  'transformation, matrix (*.npy).')

    p.add_argument('out_name', action='store', metavar='OUT_NAME',
                   type=str, help='Output filename of the transformed data.')

    p.add_argument('--inverse', action='store_true',
                   help='Will apply the inverse transformation.')

    p.add_argument('-f', action='store_true', dest='force_overwrite',
                   help='force (overwrite output file if present)')

    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    # Check if the files exist
    if not os.path.isfile(args.transformation):
        parser.error('"{0}" must be a file!'.format(args.transformation))

    if not os.path.isfile(args.ref_file):
        parser.error('"{0}" must be a file!'.format(args.ref_file))

    if not os.path.isfile(args.in_file):
        parser.error('"{0}" must be a file!'.format(args.in_file))

    if os.path.isfile(args.out_name) and not args.force_overwrite:
        parser.error('"{0}" already exists! Use -f to overwrite it.'
                     .format(args.out_name))

    transfo = np.loadtxt(args.transformation)
    if args.inverse:
        transfo = np.linalg.inv(transfo)

    ref_name, ref_extension = split_name_with_nii(args.ref_file)
    in_name, in_extension = split_name_with_nii(args.in_file)

    if ref_extension not in ['.nii', '.nii.gz']:
        parser.error('"{0}" is in an unsupported format.'.format(args.ref_file))
    if in_extension not in ['.nii', '.nii.gz']:
        parser.error('"{0}" is in an unsupported format.'.format(args.in_file))

    transform_anatomy(transfo, args.ref_file, args.in_file,
                      args.out_name)


if __name__ == "__main__":
    main()
