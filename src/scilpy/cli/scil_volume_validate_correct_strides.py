#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect when data strides are different from [1, 2, 3] and correct them.
The script takes as input a nifti file and outputs a nifti file with the
corrected strides if needed.

Input file can be 3D or 4D. Only the first 3 dimensions are considered
for the stride correction. In the case of DWI data, we recommand to also input
the b-values and b-vectors files to correct the b-vectors accordingly. If the
--validate_bvecs is set, the script first detects sign flips and/or axes swaps
in the b-vectors from a fiber coherence index [1] and corrects the b-vectors.
Then, the b-vectors are permuted and sign flipped to match the new strides.

A typical pipeline could be:
>>> scil_volume_validate_correct_strides t1.nii.gz t1_restride.nii.gz
>>> scil_volume_validate_correct_strides dwi.nii.gz dwi_restride.nii.gz
    --in_bvec dwi.bvec --in_bval dwi.bval --out_bvec dwi_restride.bvec

------------------------------------------------------------------------------
Reference:
[1] Schilling KG, Yeh FC, Nath V, Hansen C, Williams O, Resnick S, Anderson AW,
    Landman BA. A fiber coherence index for quality control of B-table
    orientation in diffusion MRI scans. Magn Reson Imaging. 2019 May;58:82-89.
    doi: 10.1016/j.mri.2019.01.018.
------------------------------------------------------------------------------
"""

import argparse
import logging

from dipy.io.gradients import read_bvals_bvecs
import numpy as np
import nibabel as nib

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg)
from scilpy.reconst.fiber_coherence import compute_coherence_table_for_transforms
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_data', 
                   help='Path to input nifti file.')
    
    p.add_argument('out_data',
                   help='Path to output nifti file with corrected strides.')
    
    p.add_argument('--in_bvec',
                   help='Path to bvec file (FSL format). If provided, the '
                        'bvecs will be permuted and sign flipped to match the '
                        'new strides.')
    
    p.add_argument('--in_bval',
                   help='Path to bval file. Must be provided if --in_bvec is '
                        'used.')
    
    p.add_argument('--out_bvec',
                   help='Path to output bvec file (FSL format). Must be '
                        'provided if --in_bvec is used.')
    
    p.add_argument('--validate_bvecs', action='store_true',
                   help='If set, the script first detects sign flips and/or '
                        'axes swaps in the b-vectors from a fiber coherence '
                        'index [1] and corrects the b-vectors before '
                        'permuting/sign flipping them to match the new '
                        'strides. If not set, the b-vectors are only permuted '
                        'and sign flipped to match the new strides.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_data],
                        optional=[args.in_bvec, args.in_bval])
    assert_outputs_exist(parser, args, args.out_data, optional=args.out_bvec)

    img = nib.load(args.in_data)
    data = img.get_fdata()
    strides = nib.io_orientation(img.affine)
    strides = strides[0] * strides[1]
    print(strides)


if __name__ == "__main__":
    main()
