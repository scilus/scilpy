#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Concatenate DWI, bval and bvecs together. File must be specified in matching
order. Default data type will be the same as the first input DWI.

Formerly: scil_concatenate_dwi.py
"""

import argparse
import logging

from dipy.io.gradients import read_bvals_bvecs
from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('out_dwi',
                   help='The name of the output DWI file.')
    p.add_argument('out_bval',
                   help='The name of the output b-values file (.bval).')
    p.add_argument('out_bvec',
                   help='The name of the output b-vectors file (.bvec).')

    p.add_argument('--in_dwis', nargs='+',
                   help='The DWI file (.nii) to concatenate.')
    p.add_argument('--in_bvals', nargs='+',
                   help='The b-values files in FSL format (.bval).')
    p.add_argument('--in_bvecs', nargs='+',
                   help='The b-vectors files in FSL format (.bvec).')

    p.add_argument('--data_type',
                   help='Data type of the output image. Use the format: '
                        'uint8, int16, int/float32, int/float64.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if len(args.in_dwis) != len(args.in_bvals) \
            or len(args.in_dwis) != len(args.in_bvecs):
        parser.error('DWI, bvals and bvecs must have the same length')

    assert_inputs_exist(parser, args.in_dwis + args.in_bvals + args.in_bvecs)
    assert_outputs_exist(parser, args, [args.out_dwi, args.out_bval,
                                        args.out_bvec])

    all_bvals = []
    all_bvecs = []
    total_size = 0
    for i in range(len(args.in_dwis)):
        bvals, bvecs = read_bvals_bvecs(args.in_bvals[i], args.in_bvecs[i])
        if len(bvals) != len(bvecs):
            raise ValueError('Paired bvals and bvecs must have the same size.')
        total_size += len(bvals)
        all_bvals.append(bvals)
        all_bvecs.append(bvecs)
    all_bvals = np.concatenate(all_bvals)
    all_bvecs = np.concatenate(all_bvecs)

    ref_dwi = nib.load(args.in_dwis[0])
    all_dwi = np.zeros(ref_dwi.shape[0:3] + (total_size,),
                       dtype=args.data_type)
    last_count = ref_dwi.shape[-1]
    all_dwi[..., 0:last_count] = ref_dwi.get_fdata()
    for i in range(1, len(args.in_dwis)):
        curr_dwi = nib.load(args.in_dwis[i])
        if not is_header_compatible(curr_dwi, ref_dwi):
            raise ValueError('All DWI must have the compatible header.')
        curr_size = curr_dwi.shape[-1]
        all_dwi[..., last_count:last_count+curr_size] = \
            curr_dwi.get_fdata()
        last_count += curr_size

    np.savetxt(args.out_bval, all_bvals, '%d')
    np.savetxt(args.out_bvec, all_bvecs.T, '%0.15f')
    nib.save(nib.Nifti1Image(all_dwi, ref_dwi.affine, header=ref_dwi.header),
             args.out_dwi)


if __name__ == "__main__":
    main()
