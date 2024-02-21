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

Formerly: scil_extract_dwi_shell.py
"""

import argparse
import logging

from dipy.io import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.dwi.utils import extract_dwi_shell
from scilpy.io.gradients import save_gradient_sampling_fsl
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
    p.add_argument('in_bvals_to_extract', nargs='+', type=int,
                   help='The list of b-values to extract. For example 0 2000.')
    p.add_argument('out_dwi',
                   help='The name of the output DWI file.')
    p.add_argument('out_bval',
                   help='The name of the output b-value file (.bval).')
    p.add_argument('out_bvec',
                   help='The name of the output b-vector file (.bvec).')

    p.add_argument('--out_indices',
                   help='Optional filename for valid indices in input dwi '
                        'volume')
    p.add_argument('--block-size', '-s', metavar='INT', type=int,
                   help='Loads the data using this block size. Useful\n'
                        'when the data is too large to be loaded in memory.')
    p.add_argument('--tolerance', '-t',
                   metavar='INT', type=int, default=20,
                   help='The tolerated gap between the b-values to  extract\n'
                        'and the actual b-values. [%(default)s]')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])
    assert_outputs_exist(parser, args, [args.out_dwi, args.out_bval,
                                        args.out_bvec], args.out_indices)

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    # Find the volume indices that correspond to the shells to extract.
    img = nib.load(args.in_dwi)

    indices, shell_data, new_bvals, new_bvecs = extract_dwi_shell(
        img, bvals, bvecs, args.in_bvals_to_extract,
        args.tolerance, args.block_size)

    logging.info("Selected indices: {}".format(indices))

    # toDo Could we use: scilpy.io.gradients.save_gradient_sampling_fsl?
    np.savetxt(args.out_bval, new_bvals, '%d')
    np.savetxt(args.out_bvec, new_bvecs.T, '%0.15f')
    nib.save(nib.Nifti1Image(shell_data, img.affine, header=img.header),
             args.out_dwi)

    # output indices file
    if args.out_indices:
        np.savetxt(args.out_indices, indices, '%u')


if __name__ == "__main__":
    main()
