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


def mask_union(left, right):
    return left + right - left * right


def mask_intersection(left, right):
    return left * right


def mask_difference(left, right):
    return left - left * right


OPERATIONS = {
    'difference': mask_difference,
    'intersection': mask_intersection,
    'union': mask_union,
}


def build_args_parser():

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION)

    p.add_argument('operation',
                   choices=list(OPERATIONS.keys()),
                   metavar='OPERATION',
                   help='The type of operation to be performed on the '
                   'masks. Must\nbe one of the following: '
                   '%(choices)s.')

    p.add_argument('inputs',
                   metavar='INPUT_FILES', nargs='+',
                   help='The list of files that contain the ' +
                   'masks to operate on. Can also be \'ones\'.')

    p.add_argument('output',
                   help='The file where the resulting mask is saved.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return parser


def main():

    parser = build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    binary_op = ['union','intersection','difference','','','']
    
    if len(img_inputs):
        assert_inputs_exist(parser, img_inputs)
    assert_outputs_exist(parser, args, args.output)

    # Load all input masks.
    masks = [load_data(f) for f in args.inputs]

    # Apply the requested operation to each input file.
    logging.info(
        'Performing operation \'{}\'.'.format(args.operation))
    mask = reduce(OPERATIONS[args.operation], masks)

    if args.threshold:
        mask = (mask > args.threshold).astype(np.uint8)

    affine = next(nibabel.load(
        f).affine for f in args.inputs if os.path.isfile(f))
    new_img = nibabel.Nifti1Image(mask, affine)
    nibabel.save(new_img, args.output)


if __name__ == "__main__":
    main()
