#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The NAWM (Normal Appearing White Matter) is the white matter that is
neighboring a lesion. It is used to compute metrics in the white matter
surrounding lesions.

This script will generate concentric rings around the lesions, with the rings
going from 2 to nb_ring + 2, with the lesion being 1.

The optional mask is used to compute the rings only in the mask
region. This can be useful to avoid useless computation.

If the lesion_atlas is binary, the output will be 3D. If the lesion_atlas
is a label map, the output will be either:
  - 4D, with each label having its own NAWM.
  - 3D, if using --split_4D and saved into a folder as multiple 3D files.

WARNING: Voxels must be isotropic.
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.image.volume_operations import compute_nawm
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             add_verbose_arg)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_image',
                   help='Lesions file as mask OR labels (.nii.gz).\n'
                        '(must be uint8 for mask, uint16 for labels).')
    p.add_argument('out_image',
                   help='Output NAWM file (.nii.gz).\n'
                        'If using --split_4D, this will be the prefix of the '
                        'output files.')

    p.add_argument('--nb_ring', type=int, default=3,
                   help='Integer representing the number of rings to be '
                        'created.')
    p.add_argument('--ring_thickness', type=int, default=2,
                   help='Integer representing the thickness (in voxels) of '
                        'the rings to be created.')
    p.add_argument('--mask',
                   help='Mask where to compute the NAWM (e.g WM mask).')
    p.add_argument('--split_4D', metavar='OUT_DIR',
                   help='Provided lesions will be split into multiple files.\n'
                        'The output files will be named using out_image as '
                        'a prefix.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if args.nb_ring < 1:
        parser.error('The number of rings must be at least 1.')
    if args.ring_thickness < 1:
        parser.error('The ring thickness must be at least 1.')

    assert_inputs_exist(parser, args.in_image, args.mask)
    if not args.split_4D:
        assert_outputs_exist(parser, args, args.out_image)

    lesion_img = nib.load(args.in_image)
    lesion_atlas = get_data_as_labels(lesion_img)
    voxel_size = lesion_img.header.get_zooms()

    if not np.allclose(voxel_size, np.mean(voxel_size)):
        raise ValueError('Voxels must be isotropic.')

    if args.split_4D and np.unique(lesion_atlas).size <= 2:
        raise ValueError('Split only works with multiple lesion labels')
    elif args.split_4D:
        assert_output_dirs_exist_and_empty(parser, args, args.split_4D)

    if not args.split_4D and np.unique(lesion_atlas).size > 2:
        logging.warning('The input lesion atlas has multiple labels. '
                        'Converting to binary.')
        lesion_atlas[lesion_atlas > 0] = 1

    if args.mask:
        mask_img = nib.load(args.mask)
        mask_data = get_data_as_mask(mask_img)
    else:
        mask_data = None

    nawm = compute_nawm(lesion_atlas, args.nb_ring, args.ring_thickness,
                        mask=mask_data)

    if args.split_4D:
        for i in range(nawm.shape[-1]):
            label = np.unique(lesion_atlas)[i+1]
            base, ext = split_name_with_nii(args.in_image)
            base = os.path.basename(base)
            lesion_name = os.path.join(args.split_4D,
                                       f'{base}_nawm_{label}{ext}')
            nib.save(nib.Nifti1Image(nawm[..., i], lesion_img.affine),
                     lesion_name)
    else:
        nib.save(nib.Nifti1Image(nawm, lesion_img.affine), args.out_image)


if __name__ == "__main__":
    main()
