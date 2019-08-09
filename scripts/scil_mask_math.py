#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from builtins import next
from past.utils import old_div
import argparse
from functools import reduce
import logging
import os

import nibabel
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_outputs_exist,
                             assert_inputs_exist)

DESCRIPTION = """
Performs an operation on a list of mask images. The supported
operations are:

    subtraction:  Keep the voxels from the first file that are not in
                  any of the following files.

    intersection: Keep the voxels that are present in all files.

    union:        Keep voxels that are in any file.

This script handles both probabilistic masks and binary masks.
"""


def mask_union(left, right):
    return left + right - left * right


def mask_intersection(left, right):
    return left * right


def mask_subtraction(left, right):
    return left - left * right


OPERATIONS = {
    'subtraction': mask_subtraction,
    'intersection': mask_intersection,
    'union': mask_union,
}


def build_args_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION)

    parser.add_argument('operation',
                        choices=list(OPERATIONS.keys()),
                        metavar='OPERATION',
                        help='The type of operation to be performed on the '
                             'masks. Must\nbe one of the following: '
                             '%(choices)s.')

    parser.add_argument('inputs',
                        metavar='INPUT_FILES', nargs='+',
                        help='The list of files that contain the ' +
                             'masks to operate on. Can also be \'ones\'.')

    parser.add_argument('output',
                        help='The file where the resulting mask is saved.')

    parser.add_argument('-t', '--threshold',
                        metavar='FLOAT', type=float,
                        help='Threshold used to transform the mask to binary.\n'
                             'Note that the threshold is applied once on the '
                             'output mask and not\nafter each operation.')

    parser.add_argument('--verbose', '-v', action='store_true', dest='verbose',
                        help='Produce verbose output.')

    add_overwrite_arg(parser)

    return parser


def load_data(path):

    if path == 'ones':
        mask = 1.0
    else:
        logging.info('Loading mask from {0}.'.format(path))
        mask = nibabel.load(path).get_data()

        min = mask.min()
        max = mask.max()
        if max == 0 and min == 0:
            logging.warning('The mask {0} is empty.'.format(path))
        elif min < 0.0 or max > 1.0:
            logging.warning(
                'The mask {0} is not binary or probabilistic. Converting it '
                'to a probabilistic mask.'.format(path))
            mask = old_div((mask.astype(float) - min), (max + min))

    return mask


def main():

    parser = build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    img_inputs = [s for s in args.inputs if s != 'ones']
    if len(img_inputs):
        assert_inputs_exist(parser, img_inputs)
    assert_outputs_exists(parser, args, [args.output])

    # Load all input masks.
    masks = [load_data(f) for f in args.inputs]

    # Apply the requested operation to each input file.
    logging.info(
        'Performing operation \'{}\'.'.format(args.operation))
    mask = reduce(OPERATIONS[args.operation], masks)

    if args.threshold:
        mask = (mask > args.threshold).astype(np.uint8)

    affine = next(nibabel.load(f).affine for f in args.inputs if os.path.isfile(f))
    new_img = nibabel.Nifti1Image(mask, affine)
    nibabel.save(new_img, args.output)


if __name__ == "__main__":
    main()
