#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split a label image into multiple images where the name of the output images
is the id of the label (ex. 35.nii.gz, 36.nii.gz, ...). If the --range option
is not provided, all labels of the image are extracted.

IMPORTANT: your label image must be of an integer type.
"""

import argparse
import os
import re

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


# Taken from http://stackoverflow.com/a/6512463
def parseNumList(str_to_parse):
    """
    Return a list of numbers from the range specified in the string.
    String to parse should be formatted as '1-3' or '3 4'.
    Example: parseNumList('2-5') == [2, 3, 4, 5]
    """
    m = re.match(r'(\d+)(?:-(\d+))?$', str_to_parse)

    if not m:
        raise argparse.ArgumentTypeError("'" + str_to_parse + "' is not a " +
                                         "range of numbers. Expected forms " +
                                         "like '1-3' or '3 4'.")

    start = m.group(1)
    end = m.group(2) or start

    start = int(start)
    end = int(end)

    if end < start:
        raise argparse.ArgumentTypeError("Range elements incorrectly " +
                                         "ordered in '" + str_to_parse + "'.")

    return list(range(start, end+1))


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('label_image',
                   help='Path of the input label file, '
                        'in a format supported by Nibabel.')
    p.add_argument('--output_dir', default='',
                   help='Put all ouptput images in a specific directory.')
    p.add_argument('--output_prefix', default='',
                   help='Prefix to be used for each output image.')

    p.add_argument('-r', '--range', type=parseNumList, nargs='*',
                   help='Specifies a subset of labels to split, '
                        'formatted as 1-3 or 3 4.')

    add_overwrite_arg(p)

    return p


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    required = args.label_image
    assert_inputs_exist(parser, required)

    label_image = nib.load(args.label_image)

    if not issubclass(label_image.get_data_dtype().type, np.integer):
        parser.error('The label image does not contain integers. ' +
                     'Will not process.\nConvert your image to integers ' +
                     'before calling.')

    label_image_data = label_image.get_fdata().astype(int)

    if args.range:
        label_indices = [item for sublist in args.range for item in sublist]
    else:
        label_indices = np.unique(label_image_data)
    label_names = [str(i) for i in label_indices]

    output_filenames = []
    for label, name in zip(label_indices, label_names):
        if int(label) != 0:
            if args.output_prefix:
                output_filenames.append(os.path.join(args.output_dir,
                                                     '{0}_{1}.nii.gz'.format(
                                                         args.output_prefix,
                                                         name)))
            else:
                output_filenames.append(os.path.join(args.output_dir,
                                                     '{0}.nii.gz'.format(
                                                        name)))

    assert_outputs_exist(parser, args, output_filenames)

    if args.output_dir and not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # Extract the voxels that match the label and save them to a file.
    cnt_filename = 0
    for label in label_indices:
        if int(label) != 0:
            split_label = np.zeros(label_image.get_header().get_data_shape(),
                                   dtype=label_image.get_data_dtype())
            split_label[np.where(label_image_data == int(label))] = label

            split_image = nib.Nifti1Image(split_label,
                                          label_image.get_affine(),
                                          label_image.get_header())
            nib.save(split_image, output_filenames[cnt_filename])
            cnt_filename += 1


if __name__ == "__main__":
    main()
