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

from scilpy.gradients.bvec_bval_tools import (flip_gradient_sampling,
                                              swap_gradient_axis)
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
                        'bvecs will \nbe permuted and sign flipped to match '
                        'the new strides.')

    p.add_argument('--out_bvec',
                   help='Path to output bvec file (FSL format). Must be '
                        'provided if --in_bvec is used.')
    
    p.add_argument('--validate_bvec', action='store_true',
                   help='If set, the script first detects sign flips and/or '
                        'axes swaps \nin the b-vectors from a fiber coherence '
                        'index [1] and corrects \nthe b-vectors before '
                        'permuting/sign flipping them to match the new '
                        'strides. \nIf not set, the b-vectors are only '
                        'permuted and sign flipped to match the new strides.')

    p.add_argument('--in_bval',
                   help='Path to bval file. Must be provided if '
                        '--validate_bvecs is used.')

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

    if args.in_bvec and not args.out_bvec:
        parser.error('--out_bvec must be provided if --in_bvec is used.')
    if args.validate_bvec and (not args.in_bvec or not args.in_bval):
        parser.error('--in_bvec and --in_bval must be provided if '
                     '--validate_bvecs is set.')

    img = nib.load(args.in_data)
    strides = nib.io_orientation(img.affine).astype(np.int8)
    strides = (strides[:, 0] + 1) * strides[:, 1]
    if np.array_equal(strides, [1, 2, 3]):
        logging.warning('Input data already has the correct strides [1, 2, 3].'
                        ' No correction needed and outputed. If you want to '
                        'validate b-vectors, please use the script '
                        'scil_gradients_validate_correct.')
        return
    else:
        logging.warning('Input data has strides {}. '
                        'Correcting to [1, 2, 3].'.format(strides))
    n = len(strides)
    transform = [0]*n
    for i, m in enumerate(strides):
        axis = abs(m) - 1
        sign = 1 if m > 0 else -1
        transform[axis] = sign * (i+1)

    axes_to_flip = []
    swapped_order = []
    for next_axis in transform:
        if next_axis in [1, 2, 3]:
            swapped_order.append(next_axis - 1)
        elif next_axis in [-1, -2, -3]:
            axes_to_flip.append(abs(next_axis) - 1)
            swapped_order.append(abs(next_axis) - 1)

    ornt = np.column_stack((np.array(swapped_order, dtype=np.int8),
                            np.where(np.isin(range(n), axes_to_flip), -1, 1)))

    new_img = img.as_reoriented(ornt)
    nib.save(new_img, args.out_data)

    if args.in_bvec and not args.validate_bvec:
        _, bvecs = read_bvals_bvecs(None, args.in_bvec)
        bvecs = flip_gradient_sampling(bvecs, axes_to_flip, 'fsl')
        bvecs = swap_gradient_axis(bvecs, swapped_order, 'fsl')
        np.savetxt(args.out_bvec, bvecs, "%.8f")


if __name__ == "__main__":
    main()
