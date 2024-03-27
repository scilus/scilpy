#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs an operation on a list of matrices. The supported operations are
listed below.

Some operations such as multiplication or addition accept float value as
parameters instead of matrices.
> scil_connectivity_math.py multiplication mat.npy 10 mult_10.npy
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.image.volume_math import (get_array_ops, get_operations_doc)
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_outputs_exist,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format)
from scilpy.utils import is_float

OPERATIONS = get_array_ops()

ADDED_DOC = get_operations_doc(OPERATIONS).replace('images', 'matrices')
ADDED_DOC = ADDED_DOC.replace('image', 'matrix')
ADDED_DOC = ADDED_DOC.replace('IMG', 'MAT')
__doc__ += ADDED_DOC


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('operation',
                   choices=OPERATIONS.keys(),
                   help='The type of operation to be performed on the '
                        'matrices.')
    p.add_argument('in_matrices', nargs='+',
                   help='The list of matrices files or parameters.')
    p.add_argument('out_matrix',
                   help='Output matrix path.')

    p.add_argument('--data_type',
                   help='Data type of the output image. Use the format: '
                        'uint8, float16, int32.')
    p.add_argument('--exclude_background', action='store_true',
                   help='Does not affect the background of the original '
                        'matrices.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def load_matrix(arg):
    if is_float(arg):
        matrix = float(arg)
    else:
        if not os.path.isfile(arg):
            raise ValueError('Input file {} does not exist.'.format(arg))

        data = load_matrix_in_any_format(arg).astype(np.float64)
        matrix = nib.Nifti1Image(data, np.eye(4))
        logging.info('Loaded {} of shape {} and data_type {}.'.format(
            arg, data.shape, data.dtype))

        if data.ndim > 2:
            logging.warning('{} has {} dimensions, be careful.'.format(
                arg, data.ndim))
        elif data.ndim < 2:
            raise ValueError('{} has {} dimensions, not valid.'.format(
                arg, data.ndim))

    return matrix


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_outputs_exist(parser, args, args.out_matrix)

    # Binary operations require specific verifications
    binary_op = ['union', 'intersection', 'difference', 'invert']

    if args.operation not in OPERATIONS.keys():
        parser.error('Operation {} not implemented.'.format(args.operation))

    # Find at least one matrix for reference
    for input_arg in args.in_matrices:
        found_ref = False
        if not is_float(input_arg):
            ref_data = load_matrix_in_any_format(input_arg)
            ref_matrix = nib.Nifti1Image(ref_data, np.eye(4))
            mask = np.zeros(ref_data.shape)
            found_ref = True
            break

    if not found_ref:
        raise ValueError('Requires at least one matrix.')

    # Load all input matrices
    input_matrices = []
    for input_arg in args.in_matrices:
        matrix = load_matrix(input_arg)

        if args.operation in binary_op and isinstance(matrix, nib.Nifti1Image):
            data = matrix.get_fdata(dtype=np.float64)
            unique = np.unique(data)
            if not len(unique) <= 2:
                parser.error('Binary operations can only be performed with '
                             'binary masks.')

            if len(unique) == 2 and not (unique == [0, 1]).all():
                logging.warning('Input data for binary operation are not '
                                'binary array, will be converted.\n'
                                'Non-zeros will be set to ones.')
                data[data != 0] = 1

        if isinstance(matrix, nib.Nifti1Image):
            data = matrix.get_fdata(dtype=np.float64)
            mask[data > 0] = 1
        input_matrices.append(matrix)

    if args.operation == 'convert' and not args.data_type:
        parser.error('Convert operation must be used with --data_type.')

    # Perform the request operation
    try:
        output_data = OPERATIONS[args.operation](input_matrices, ref_matrix)
    except ValueError:
        logging.error('{} operation failed.'.format(
            args.operation.capitalize()))
        return

    # Cast if needed
    if args.data_type:
        output_data = output_data.astype(args.data_type)
    else:
        output_data = output_data.astype(np.float64)

    if args.exclude_background:
        output_data[mask == 0] = 0

    # Saving in the right format
    save_matrix_in_any_format(args.out_matrix, output_data)


if __name__ == "__main__":
    main()
