#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extracts the DWI volumes that are on specific b-value shells. Many shells
can be extracted at once by specifying multiple b-values. The extracted
volumes are in the same order as in the original file.

If the b-values of a shell are not all identical, use the --tolerance
argument to adjust the accepted interval. For example, a b-value of 2000
and a tolerance of 20 will extract all volumes with a b-values from 1980 to
2020.

Files that are too large to be loaded in memory can still be processed by
setting the --block-size argument. A block size of X means that X DWI volumes
are loaded at a time for processing.

"""

import argparse
import logging

from dipy.io import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.utils.bvec_bval_tools import extract_dwi_shell


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('dwi',
                        help='The DW image file to split.')

    parser.add_argument('bvals',
                        help='The b-values in FSL format.')

    parser.add_argument('bvecs',
                        help='The b-vectors in FSL format.')

    parser.add_argument('bvals_to_extract', nargs='+',
                        metavar='bvals-to-extract', type=int,
                        help='The list of b-values to extract. For example '
                             '0 2000.')

    parser.add_argument('output_dwi',
                        help='The name of the output DWI file.')

    parser.add_argument('output_bvals',
                        help='The name of the output b-values.')

    parser.add_argument('output_bvecs',
                        help='The name of the output b-vectors')

    parser.add_argument('--block-size', '-s',
                        metavar='INT', type=int,
                        help='Loads the data using this block size. '
                             'Useful\nwhen the data is too large to be '
                             'loaded in memory.')

    parser.add_argument('--tolerance', '-t',
                        metavar='INT', type=int, default=20,
                        help='The tolerated gap between the b-values to '
                             'extract\nand the actual b-values.')

    add_verbose_arg(parser)
    add_overwrite_arg(parser)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.dwi, args.bvals, args.bvecs])
    assert_outputs_exist(parser, args, [args.output_dwi, args.output_bvals,
                                        args.output_bvecs])

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    # Find the volume indices that correspond to the shells to extract.
    tol = args.tolerance

    img = nib.load(args.dwi)

    outputs = extract_dwi_shell(img, bvals, bvecs, args.bvals_to_extract, tol,
                                args.block_size)
    indices, shell_data, new_bvals, new_bvecs = outputs

    logging.info("Selected indices: {}".format(indices))

    np.savetxt(args.output_bvals, new_bvals, '%d')
    np.savetxt(args.output_bvecs, new_bvecs.T, '%0.15f')
    nib.save(nib.Nifti1Image(shell_data, img.affine, img.header),
             args.output_dwi)


if __name__ == "__main__":
    main()
