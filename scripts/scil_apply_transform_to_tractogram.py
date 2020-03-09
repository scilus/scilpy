#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Transform tractogram using an affine/rigid transformation.

    For more information on how to use the various registration scripts
    see the doc/tractogram_registration.md readme file
"""

import argparse

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, create_header_from_anat,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.utils.filenames import split_name_with_nii


def transform_tractogram(in_filename, ref_filename, transfo,
                         filename_to_save):
    tractogram = nib.streamlines.load(in_filename)

    _, out_extension = split_name_with_nii(filename_to_save)
    if out_extension == '.trk':
        # Only TRK/NII can be a reference, because they have an affine
        _, ref_extension = split_name_with_nii(ref_filename)
        if ref_extension == '.trk':
            ref_tractogram = nib.streamlines.load(ref_filename, lazy_load=True)
            ref_header = ref_tractogram.header
        else:
            ref_img = nib.load(ref_filename)
            ref_header = create_header_from_anat(ref_img)
    elif out_extension == '.tck':
        ref_header = nib.streamlines.TckFile.create_empty_header()

    tractogram.tractogram.apply_affine(transfo)

    new_tractogram = nib.streamlines.Tractogram(tractogram.streamlines,
                                                affine_to_rasmm=np.eye(4))
    nib.streamlines.save(new_tractogram, filename_to_save, header=ref_header)


def _build_args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_file',
                   help='Path of the tractogram to be transformed (trk or tck).')

    p.add_argument('ref_file',
                   help='Path of the reference file (trk or nii)')

    p.add_argument('transformation',
                   help='Path of the file containing the 4x4 \n'
                   'transformation, matrix (*.npy).'
                   'See the script description for more information.')

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

    _, ref_extension = split_name_with_nii(args.ref_file)
    if ref_extension == '.trk':
        if not nib.streamlines.TrkFile.is_correct_format(args.ref_file):
            parser.error('{} is not a valid TRK file.'.format(args.ref_file))
    elif ref_extension not in ['.nii', '.nii.gz']:
        parser.error('{} is an unsupported format.'.format(args.ref_file))

    transfo = np.loadtxt(args.transformation)
    if args.inverse:
        transfo = np.linalg.inv(transfo)

    transform_tractogram(args.in_file, args.ref_file, transfo,
                         args.out_name)


if __name__ == "__main__":
    main()
