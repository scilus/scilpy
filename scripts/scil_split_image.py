#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Splits the DWI image at certain indices along the last dimension (b-values).
Many indices can be given at once by specifying multiple values. The splited
volumes are in the same order as in the original file. Also outputs the
corresponding .bval and .bvec files.

This script can be useful for splitting images at places where a b-value
extraction does not work. For instance, if one wants to split the x first
b-1500s from the rest of the b-1500s in an image, simply put x as an index.

"""

import argparse
import logging
from pathlib import Path

from dipy.io import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dwi',
                   help='The DW image file to split.')

    p.add_argument('in_bval',
                   help='The b-values file in FSL format (.bval).')

    p.add_argument('in_bvec',
                   help='The b-vectors file in FSL format (.bvec).')

    p.add_argument('split_indices', nargs='+', type=int,
                   help='The list of indices where to split the image. For '
                        'example 3 10. This would split the image in three '
                        'parts, such as [:3], [3:10], [10:].')

    p.add_argument('--out_dwi', nargs='+', default=[],
                   help='The names of the output DWI files. There must be '
                        'one more name than indices in split_indices. By '
                        'default, indices number will be appended to in_dwi, '
                        'such as in_dwi_0_3, in_dwi_3_10, in_dwi_10_end.')

    p.add_argument('--out_bval', nargs='+', default=[],
                   help='The names of the output b-value files (.bval). There '
                        'must be one more name than indices in split_indices. '
                        'By default, indices number will be appended to '
                        'in_dwi, such as in_dwi_0_3, in_dwi_3_10, '
                        'in_dwi_10_end.')

    p.add_argument('--out_bvec', nargs='+', default=[],
                   help='The names of the output b-vector files (.bvec).There '
                        'must be one more name than indices in split_indices. '
                        'By default, indices number will be appended to '
                        'in_dwi, such as in_dwi_0_3, in_dwi_3_10, '
                        'in_dwi_10_end.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def create_name_from_indices(indices, input):
    input_path = Path(input)
    parent_dir = input_path.parent
    input_name = Path(input_path.stem).stem
    index_name = "_" + str(indices[0]) + "_" + str(indices[1])
    suffix = ""
    for suff in input_path.suffixes:
        suffix += suff
    output_name = Path(parent_dir, input_name + index_name + suffix)

    return str(output_name)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])
    assert_outputs_exist(parser, args, [],
                         optional=list(np.concatenate((args.out_dwi,
                                                       args.out_bval,
                                                       args.out_bvec))))

    # Check if the number of names given is equal to the number of indices + 1
    if args.out_dwi and len(args.out_dwi) != len(args.split_indices) + 1:
        parser.error('--out_dwi must contain len(split_indices) + 1 names.')
    if args.out_bval and len(args.out_bval) != len(args.split_indices) + 1:
        parser.error('--out_bval must contain len(split_indices) + 1 names.')
    if args.out_bvec and len(args.out_bvec) != len(args.split_indices) + 1:
        parser.error('--out_bvec must contain len(split_indices) + 1 names.')

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    img = nib.load(args.in_dwi)
    data = img.get_fdata(dtype=np.float32)

    # Check if the indices fit inside the range of possible values
    if np.max(args.split_indices) > data.shape[-1]:
        parser.error('split_indices values must be lower than the total '
                     'number of direcitons.')
    if np.min(args.split_indices) <= 0:
        parser.error('split_indices values must be higher than 0.')

    indices = np.concatenate(([0], args.split_indices, [data.shape[-1]]))

    for i in range(len(indices)-1):
        data_split = data[..., indices[i]:indices[i+1]]
        bvals_split = bvals[indices[i]:indices[i+1]]
        bvecs_split = bvecs[indices[i]:indices[i+1]]
        # Saving the dwi file
        if args.out_dwi:
            data_name = args.out_dwi[i]
        else:
            data_name = create_name_from_indices(indices[i:i+2], args.in_dwi)
        nib.save(nib.Nifti1Image(data_split, img.affine, header=img.header),
                 data_name)
        # Saving the bval file
        if args.out_bval:
            bval_name = args.out_bval[i]
        else:
            bval_name = create_name_from_indices(indices[i:i+2], args.in_bval)
        np.savetxt(bval_name, bvals_split, '%d')
        # Saving the bvec file
        if args.out_bvec:
            bvec_name = args.out_bvec[i]
        else:
            bvec_name = create_name_from_indices(indices[i:i+2], args.in_bvec)
        np.savetxt(bvec_name, bvecs_split, '%0.15f')


if __name__ == "__main__":
    main()
