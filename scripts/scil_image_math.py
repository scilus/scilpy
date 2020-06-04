#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs an operation on a list of images. The supported operations are
listed below.

This script is loading all images in memory, will often crash after a few
hundred images.

Some operations such as multiplication or addition accept float value as
parameters instead of images.
> scil_image_math.py multiplication img.nii.gz 10 mult_10.nii.gz
"""

import argparse
import logging
import os

from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.image.operations import (get_image_ops, get_operations_doc)
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_outputs_exist)
from scilpy.utils.util import is_float

OPERATIONS = get_image_ops()

__doc__ += get_operations_doc(OPERATIONS)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('operation',
                   choices=OPERATIONS.keys(),
                   help='The type of operation to be performed on the '
                        'images.')
    p.add_argument('in_images', nargs='+',
                   help='The list of image files or parameters.')
    p.add_argument('out_image',
                   help='Output image path.')

    p.add_argument('--data_type',
                   help='Data type of the output image. Use the format: '
                        'uint8, int16, int/float32, int/float64.')
    p.add_argument('--exclude_background', action='store_true',
                   help='Does not affect the background of the original image.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def load_data(arg):
    if is_float(arg):
        data = float(arg)
    else:
        if not os.path.isfile(arg):
            raise ValueError('Input file {} does not exist.'.format(arg))
        data = np.asanyarray(nib.load(arg).dataobj)
        logging.info('Loaded {} of shape {} and data_type {}.'.format(
                     arg, data.shape, data.dtype))

        if data.ndim > 3:
            logging.warning('{} has {} dimensions, be careful.'.format(
                arg, data.ndim))
        elif data.ndim < 3:
            raise ValueError('{} has {} dimensions, not valid.'.format(
                arg, data.ndim))

    return data


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_outputs_exist(parser, args, args.out_image)

    # Binary operations require specific verifications
    binary_op = ['union', 'intersection', 'difference', 'invert',
                 'dilation', 'erosion', 'closing', 'opening']

    if args.operation not in OPERATIONS.keys():
        parser.error('Operation {} not implement.'.format(args.operation))

    # Find at least one image for reference
    for input_arg in args.in_images:
        if not is_float(input_arg):
            ref_img = nib.load(input_arg)
            mask = np.zeros(ref_img.shape)
            break

    # Load all input masks.
    input_data = []
    for input_arg in args.in_images:
        if not is_float(input_arg) and \
                not is_header_compatible(ref_img, input_arg):
            parser.error('Inputs do not have a compatible header.')
        data = load_data(input_arg)

        if isinstance(data, np.ndarray) and \
            data.dtype != ref_img.get_data_dtype() and \
                not args.data_type:
            parser.error('Inputs do not have a compatible data type.\n'
                         'Use --data_type to specify output datatype.')
        if args.operation in binary_op and isinstance(data, np.ndarray):
            unique = np.unique(data)
            if not len(unique) <= 2:
                parser.error('Binary operations can only be performed with '
                             'binary masks')

            if len(unique) == 2 and not (unique == [0, 1]).all():
                logging.warning('Input data for binary operation are not '
                                'binary arrays, will be converted.\n'
                                'Non-zeros will be set to ones.')
                data[data != 0] = 1

        if isinstance(data, np.ndarray):
            data = data.astype(np.float64)
            mask[data > 0] = 1
        input_data.append(data)

    if args.operation == 'convert' and not args.data_type:
        parser.error('Convert operation must be used with --data_type.')

    try:
        output_data = OPERATIONS[args.operation](input_data)
    except ValueError:
        logging.error('{} operation failed.'.format(
            args.operation.capitalize()))
        return

    if args.data_type:
        output_data = output_data.astype(args.data_type)
        ref_img.header.set_data_dtype(args.data_type)
    else:
        output_data = output_data.astype(ref_img.get_data_dtype())

    if args.exclude_background:
        output_data[mask == 0] = 0

    new_img = nib.Nifti1Image(output_data, ref_img.affine,
                              header=ref_img.header)
    nib.save(new_img, args.out_image)


if __name__ == "__main__":
    main()
