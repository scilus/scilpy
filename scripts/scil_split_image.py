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
                        'parts, such as [:3], [3:10], [10:]. Indices must be '
                        'in increasing order.')

    p.add_argument('--out_basename', nargs='+', default=[],
                   help='The basenames of the output files. There must be '
                        'one more name than indices in split_indices. By '
                        'default, indices number will be appended to in_dwi, '
                        'such as in_dwi_0_3, in_dwi_3_10, in_dwi_10_end.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])
    assert_outputs_exist(parser, args, [],
                         optional=list((args.out_basename)))

    # Check if the number of names given is equal to the number of indices + 1
    if (args.out_basename and 
        len(args.out_basename) != len(args.split_indices) + 1):
        parser.error('--out_basename must contain len(split_indices) + 1 '
                     'names.')

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    img = nib.load(args.in_dwi)

    # Check if the indices fit inside the range of possible values
    if np.max(args.split_indices) > img.shape[-1]:
        parser.error('split_indices values must be lower than the total '
                     'number of direcitons.')
    if np.min(args.split_indices) <= 0:
        parser.error('split_indices values must be higher than 0.')
    # Check if the indices are in increasing order
    if not np.all(np.diff(args.split_indices) > 0):
        parser.error('split_indices values must be in increasing order.')

    indices = np.concatenate(([0], args.split_indices, [img.shape[-1]]))

    for i in range(len(indices)-1):
        data_split = img.dataobj[..., indices[i]:indices[i+1]]
        bvals_split = bvals[indices[i]:indices[i+1]]
        bvecs_split = bvecs[indices[i]:indices[i+1]]
        # Saving the output files
        if args.out_basename:
            data_name = args.out_basename[i]
            bval_name = args.out_basename[i]
            bvec_name = args.out_basename[i]
        else:
            index_name = "_" + str(indices[i]) + "_" + str(indices[i+1])
            data_name = str(Path(Path(args.in_dwi).stem).stem) + index_name
            bval_name = str(Path(Path(args.in_bval).stem).stem) + index_name
            bvec_name = str(Path(Path(args.in_bvec).stem).stem) + index_name
        nib.save(nib.Nifti1Image(data_split, img.affine, header=img.header),
                 data_name + ".nii.gz")
        np.savetxt(bval_name + ".bval", bvals_split, '%d')
        np.savetxt(bvec_name + ".bvec", bvecs_split, '%0.15f')


if __name__ == "__main__":
    main()
