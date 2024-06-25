#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs an operation on a list of images. The supported operations are
listed below.

This script is loading all images in memory, will often crash after a few
hundred images.

Some operations such as multiplication or addition accept float value as
parameters instead of images.
> scil_volume_math.py multiplication img.nii.gz 10 mult_10.nii.gz

Formerly: scil_image_math.py
"""

import argparse
import logging

from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.image.volume_math import (get_image_ops, get_operations_doc)
from scilpy.io.image import load_img
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_outputs_exist)
from scilpy.utils import is_float

OPERATIONS = get_image_ops()

__doc__ += get_operations_doc(OPERATIONS)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('operation',
                   choices=OPERATIONS.keys(),
                   help='The type of operation to be performed on the images.')
    p.add_argument('in_args', nargs='+',
                   help="The list of image files or parameters. Refer to each "
                        "operation's documentation of the expected arguments.")
    p.add_argument('out_image',
                   help='Output image path.')

    p.add_argument('--data_type',
                   help='Data type of the output image. Use the format: \n'
                        'uint8, int16, int/float32, int/float64.')
    p.add_argument('--exclude_background', action='store_true',
                   help='Does not affect the background of the original '
                        'images.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_outputs_exist(parser, args, args.out_image)

    # Binary operations require specific verifications
    binary_op = ['union', 'intersection', 'difference', 'invert',
                 'dilation', 'erosion', 'closing', 'opening']

    if args.operation not in OPERATIONS.keys():
        parser.error('Operation {} not implemented.'.format(args.operation))

    # Find at least one image for reference
    # Find at least one mask, but prefer a 4D mask if there is any.
    mask = None
    found_ref = False
    for input_arg in args.in_args:
        if not is_float(input_arg):
            ref_img = nib.load(input_arg)
            found_ref = True
            if mask is None:
                mask = np.zeros(ref_img.shape)
            elif len(ref_img.shape) == 4:
                mask = np.zeros(ref_img.shape)
                break

    if not found_ref:
        raise ValueError('Requires at least one nifti image.')

    # Load all input masks.
    input_img = []
    for input_arg in args.in_args:
        if not is_float(input_arg) and \
                not is_header_compatible(ref_img, input_arg):
            parser.error('Inputs do not have a compatible header.')
        img, dtype = load_img(input_arg)

        if isinstance(img, nib.Nifti1Image) and \
            dtype != ref_img.get_data_dtype() and \
                not args.data_type:
            parser.error('Inputs do not have a compatible data type.\n'
                         'Use --data_type to specify output datatype.')
        if args.operation in binary_op and isinstance(img, nib.Nifti1Image):
            data = img.get_fdata(dtype=np.float64)
            unique = np.unique(data)
            if not len(unique) <= 2:
                parser.error('Binary operations can only be performed with '
                             'binary masks')

            if len(unique) == 2 and not (unique == [0, 1]).all():
                logging.warning('Input data for binary operation are not '
                                'binary arrays, will be converted.\n'
                                'Non-zeros will be set to ones.')

        if isinstance(img, nib.Nifti1Image):
            data = img.get_fdata(dtype=np.float64)
            if data.ndim == 4:
                mask[np.sum(data, axis=3).astype(bool) > 0] = 1
            else:
                mask[data > 0] = 1
            img.uncache()
        input_img.append(img)

    if args.operation == 'convert' and not args.data_type:
        parser.error('Convert operation must be used with --data_type.')

    try:
        output_data = OPERATIONS[args.operation](input_img, ref_img)
    except ValueError as msg:
        logging.error('{} operation failed.'.format(
            args.operation.capitalize()))
        logging.error(msg)
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
