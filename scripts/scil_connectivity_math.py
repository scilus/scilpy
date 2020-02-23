#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os

from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.image.operations import (is_float, absolute_value, around,
                                     addition, ceil, convert,
                                     difference, division, floor, intersection,
                                     invert, lower_clip, lower_threshold, mean,
                                     multiplication, normalize_max,
                                     normalize_sum, std, subtraction,
                                     union, upper_threshold, upper_clip,
                                     get_array_operation_doc, 
                                     get_image_operation_doc)
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_outputs_exist)

OPERATIONS = {
    'lower_threshold': lower_threshold,
    'upper_threshold': upper_threshold,
    'lower_clip': lower_clip,
    'upper_clip': upper_clip,
    'absolute_value': absolute_value,
    'round': around,
    'ceil': ceil,
    'floor': floor,
    'normalize_sum': normalize_sum,
    'normalize_max': normalize_max,
    'convert': convert,
    'invert': invert,
    'addition': addition,
    'subtraction': subtraction,
    'multiplication': multiplication,
    'division': division,
    'mean': mean,
    'std': std,
    'union': union,
    'intersection': intersection,
    'difference': difference,
}

DESCRIPTION = """
Performs an operation on a list of matrices. The supported operations are 
listed below.

Some operations such as multiplication or addition accept float value as
parameters instead of matrices.
> scil_connectivity_math.py multiplication mat.npy 10 mat mult_10.npy
"""

ADDED_DOC = get_array_operation_doc().replace('images', 'matrices')
ADDED_DOC = ADDED_DOC.replace('image', 'matrix')
ADDED_DOC = ADDED_DOC.replace('IMG', 'MAT')
DESCRIPTION += ADDED_DOC


def build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION)

    p.add_argument('operation',
                   help='The type of operation to be performed on the '
                   'matrices.')

    p.add_argument('inputs', nargs='+',
                   help='The list of matrices files or parameters.')

    p.add_argument('--data_type',
                   help='Data type of the output matrix.')

    p.add_argument('output',
                   help='Output matrix path.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def load_data(arg):
    if is_float(arg):
        data = float(arg)
    else:
        if not os.path.isfile(arg):
            logging.error('Input file %s does not exist', arg)
            raise ValueError

        _, ext = os.path.splitext(arg)
        if ext == '.txt':
            data = np.loadtxt(arg)
        elif ext == '.npy':
            data = np.load(arg)
        else:
            logging.error('Extension {} is not supported'.format(ext))
            raise ValueError
        logging.info('Loaded %s of shape %s and data_type %s',
                     arg, data.shape, data.dtype)

        if data.ndim > 2:
            logging.warning('%s has %s dimensions, be careful', arg, data.ndim)
        elif data.ndim < 2:
            logging.warning('%s has %s dimensions, not valid ', arg, data.ndim)
            raise ValueError

    return data


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_outputs_exist(parser, args, args.output)

    # Binary operations require specific verifications
    binary_op = ['union', 'intersection', 'difference', 'invert']

    if args.operation not in OPERATIONS.keys():
        parser.error('Operation {} not implement'.format(args.operation))

    # Load all input masks.
    input_data = []
    for input_arg in args.inputs:
        data = load_data(input_arg)

        if args.operation in binary_op and isinstance(data, np.ndarray):
            unique = np.unique(data)
            if not len(unique) <= 2:
                parser.error('Binary operations can only be performed with '
                             'binary masks')

            if len(unique) == 2 and not (unique == [0, 1]).all():
                logging.warning('Input data for binary operation are not '
                                'binary array, will be converted. '
                                'Non-zeros will be set to ones.')
                data[data != 0] = 1

        if isinstance(data, np.ndarray):
            data = data.astype(np.float64)
        input_data.append(data)

    if args.operation == 'convert' and not args.data_type:
        parser.error('Convert operation must be used with --data_type')

    output_data = OPERATIONS[args.operation](input_data)

    if args.data_type:
        output_data = output_data.astype(args.data_type)
    else:
        output_data = output_data.astype(np.float64)
    
    _, ext = os.path.splitext(args.output)
    if ext == '.txt':
        np.savetxt(args.output, output_data)
    elif ext == '.npy':
        np.save(args.output, output_data)
    else:
        parser.error('Extension {} is not supported'.format(ext))


if __name__ == "__main__":
    main()
