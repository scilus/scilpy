#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from copy import copy
import logging
from numbers import Number
import os

from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_outputs_exist,
                             assert_inputs_exist)

DESCRIPTION = """
Performs an operation on a list of mask images. The supported
operations are:

    difference:  Keep the voxels from the first file that are not in
                 any of the following files.

    intersection: Keep the voxels that are present in all files.

    union:        Keep voxels that are in any file.

This script handles both probabilistic masks and binary masks.
"""


def build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION)

    p.add_argument('operation',
                   help='The type of operation to be performed on the '
                   'masks. Must\nbe one of the following: '
                   '%(choices)s.')

    p.add_argument('inputs', nargs='+',
                   help='The list of files that contain the ' +
                   'masks to operate on. Can also be \'ones\'.')

    p.add_argument('--data_type',
                   help='Data type.')

    p.add_argument('output',
                   help='The file where the resulting mask is saved.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def is_float(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


def load_data(arg):
    if is_float(arg):
        mask = float(arg)
    else:
        mask = nib.load(arg).get_data()

        if mask.ndim > 3:
            logging.warning('%s has %s dimensions, be careful')

    return mask


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Binary operations require specific verifications
    binary_op = ['union', 'intersection', 'difference', 'inverse']

    # Single image and one value
    single_img_op_with_params = ['lower_threshold', 'upper_threshold',
                                 'lower_clip', 'upper_clip']

    # Single image and no value. Last one is a binary operation
    single_img_op_no_params = ['absolute_value', 'round', 'ceil', 'floor',
                               'normalize_sum', 'normalize_max',
                               'convert', 'inverse']

    # Last three are binary operations
    multi_img_op = ['addition', 'subtraction',
                    'multiplication', 'division',
                    'mean', 'std',
                    'union', 'intersection', 'difference']

    if args.operation not in single_img_op_with_params + \
            single_img_op_no_params + \
            multi_img_op:
        parser.error('Operation %s not implement', args.operation)

    at_least_one_img = False
    for input_arg in args.inputs:
        try:
            ref_img = nib.load(input_arg)
            at_least_one_img = True
        except:
            continue

    if not at_least_one_img:
        parser.error('At least one input should be an image')

    if args.operation in single_img_op_with_params and \
            not len(args.inputs) == 2 and isinstance(args.inputs[1], Number):
        parser.error('Selected operations only accept one image '
                     'and one number')
    elif args.operation in multi_img_op:
        if not len(args.inputs) > 1:
            parser.error('Selected operations required at least two input')
        if args.operation in ['division', 'diffrence', 'subtraction'] and \
                not len(args.inputs) == 2:
            parser.error('Operation %s only support two inputs',
                         args.operation)

    # Load all input masks.
    input_data = []
    for input_arg in args.inputs:
        if not is_float(input_arg) and \
                not is_header_compatible(ref_img, input_arg):
            parser.error('Input do not have a compatible header')
        data = load_data(input_arg)
        if isinstance(data, np.ndarray) and data.dtype != ref_img.get_data_dtype():
            parser.error('Input do not have a compatible data type.'
                         'Use --data_type to specified output datatype.')
        if args.operation in binary_op and isinstance(data, np.ndarray):
            if not len(np.unique(data)) == 2:
                parser.error('Binary operations can only be performed with '
                             'binary masks')
            if not np.unique(data) == [0, 1]:
                logging.warning('Input data for binary operation are not'
                                'binary array, will be converted.'
                                'Non-zeros will be set to ones.')
                data[data != 0] = 1

        input_data.append(data)

    # output_data = np.zeros(ref_img.get_shape())
    if args.operation == 'lower_threshold':
        output_data = copy(input_data[0])
        output_data[input_data[0] < input_data[1]] = 0
        output_data[input_data[0] >= input_data[1]] = 1
    elif args.operation == 'upper_threshold':
        output_data = copy(input_data[0])
        output_data[input_data[0] <= input_data[1]] = 1
        output_data[input_data[0] > input_data[1]] = 0
    elif args.operation == 'lower_clip':
        output_data = np.clip(input_data[0], input_data[1], None)
    elif args.operation == 'upper_clip':
        output_data = np.clip(input_data[0], None, input_data[1])
    elif args.operation == 'absolute_value':
        output_data = np.abs(input_data[0])
    elif args.operation == 'round':
        output_data = np.round(input_data[0])
    elif args.operation == 'ceil':
        output_data = np.ceil(input_data[0])
    elif args.operation == 'floor':
        output_data = np.floor(input_data[0])
    elif args.operation == 'normalize_sum':
        output_data = copy(input_data[0]) / np.sum(input_data[0])
    elif args.operation == 'normalize_max':
        output_data = copy(input_data[0]) / np.max(input_data[0])
    elif args.operation == 'convert':
        output_data = copy(input_data[0])
    elif args.operation == 'addition':
        output_data = np.zeros(ref_img.get_shape())
        for data in input_data:
            output_data += data
    elif args.operation == 'subtraction':
        output_data = np.zeros(ref_img.get_shape())
        for data in input_data:
            output_data -= data
        output_data = copy(input_data[0]) / input_data[1]
    elif args.operation == 'mean':
        output_data = np.average(input_data)
    elif args.operation == 'std':
        output_data = np.std(input_data)
    elif args.operation == 'union':
        output_data = np.zeros(ref_img.get_shape())
        for data in input_data:
            output_data += data
        output_data[output_data != 0] = 1
    elif args.operation == 'intersection':
        output_data = np.zeros(ref_img.get_shape())
        for data in input_data:
            output_data *= data
        output_data[output_data != 0] = 1
    elif args.operation == 'difference':
        output_data = copy(input_data[0]).astype(np.bool)
        output_data[input_data[1] != 0] = 0
    elif args.operation == 'inverse':
        output_data = np.zeros(ref_img.get_shape())
        output_data[input_data[0] != 0] = 0
        output_data[input_data[0] == 0] = 1

    if args.data_type:
        output_data = output_data.astype(args.data_type)
        ref_img.header.set_data_dtype(args.data_type)
    else:
        output_data = output_data.astype(ref_img.get_data_dtype())
    new_img = nib.Nifti1Image(output_data, ref_img.affine,
                              header=ref_img.header)
    nib.save(new_img, args.output)


if __name__ == "__main__":
    main()
