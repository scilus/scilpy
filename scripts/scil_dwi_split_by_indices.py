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

Formerly: scil_split_image.py
"""

import argparse
import logging

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

    p.add_argument('out_basename',
                   help='The basename of the output files. Indices number '
                        'will be appended to out_basename. For example, if '
                        'split_indices were 3 10, the files would be saved as '
                        'out_basename_0_2, out_basename_3_10, '
                        'out_basename_11_20, where the size of the last '
                        'dimension is 21 in this example.')

    p.add_argument('split_indices', nargs='+', type=int,
                   help='The list of indices where to split the image. For '
                        'example 3 10. This would split the image in three '
                        'parts, such as [:3], [3:10], [10:]. Indices must be '
                        'in increasing order.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    img = nib.load(args.in_dwi)

    # Check if the indices fit inside the range of possible values
    if np.max(args.split_indices) >= img.shape[-1]:
        parser.error('split_indices values must be lower than the total '
                     'number of directions.')
    if np.min(args.split_indices) <= 0:
        parser.error('split_indices values must be higher than 0.')
    # Check if the indices are in increasing order
    if not np.all(np.diff(args.split_indices) > 0):
        logging.warning('split_indices values were not in increasing order.'
                        'Proceeding to reorder them.')
        split_indices = np.sort(args.split_indices)
    else:
        split_indices = args.split_indices

    indices = np.concatenate(([0], split_indices, [img.shape[-1]]))

    out_names = np.ndarray((len(split_indices) + 1), dtype=object)
    for i in range(len(indices)-1):
        index_name = "_" + str(indices[i]) + "_" + str(indices[i+1] - 1)
        out_names[i] = args.out_basename + index_name
    assert_outputs_exist(parser, args, out_names)

    for i in range(len(indices)-1):
        data_split = img.dataobj[..., indices[i]:indices[i+1]]
        bvals_split = bvals[indices[i]:indices[i+1]]
        bvecs_split = bvecs[indices[i]:indices[i+1]]
        # Saving the output files
        nib.save(nib.Nifti1Image(data_split, img.affine, header=img.header),
                 out_names[i] + ".nii.gz")
        np.savetxt(out_names[i] + ".bval", bvals_split, '%d')
        np.savetxt(out_names[i] + ".bvec", bvecs_split, '%0.15f')


if __name__ == "__main__":
    main()
