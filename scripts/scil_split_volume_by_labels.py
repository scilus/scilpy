#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import inspect
import json
import os

import nibabel as nib
import numpy as np

import scilpy
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)

DESCRIPTION = """
Split a label image into multiple images where the name of the output images
is taken from a lookup table (ex: left-lateral-occipital.nii.gz,
right-thalamus.nii.gz, ...). Only the labels included in the lookup table
are extracted.

IMPORTANT: your label image must be of an integer type.
"""


def _build_args_parser(luts):

    p = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('label_image',
                   help='Path of the input label file, '
                        'in a format supported by Nibabel.')
    p.add_argument('--output_dir', default='',
                   help='Put all ouptput images in a specific directory.')
    p.add_argument('--output_prefix', default='',
                   help='Prefix to be used for each output image.')

    mutual_group = p.add_mutually_exclusive_group(required=True)
    mutual_group.add_argument(
        '--scilpy_lut', choices=luts,
        help='Lookup table, in the file scilpy/data/LUT, '
             'used to name the output files.')
    mutual_group.add_argument(
        '--custom_lut',
        help='Path of the lookup table file, '
             'used to name the output files.')

    add_overwrite_arg(p)

    return p


def main():
    # Get the valid LUT choices.
    module_path = inspect.getfile(scilpy)

    lut_dir = os.path.join(os.path.dirname(
                           os.path.dirname(module_path)) + "/data/LUT/")

    luts = [os.path.splitext(f)[0] for f in os.listdir(lut_dir)]

    parser = _build_args_parser(luts)
    args = parser.parse_args()

    required = args.label_image
    assert_inputs_exist(parser, required)

    label_image = nib.load(args.label_image)

    if not issubclass(label_image.get_data_dtype().type, np.integer):
        parser.error('The label image does not contain integers. ' +
                     'Will not process.\nConvert your image to integers ' +
                     'before calling.')

    label_image_data = label_image.get_fdata().astype(int)

    if args.scilpy_lut is not None:
        with open(os.path.join(lut_dir, args.scilpy_lut + '.json')) as f:
            label_dict = json.load(f)
        (label_indices, label_names) = zip(*label_dict.items())
    else:
        with open(args.custom_lut) as f:
            label_dict = json.load(f)
        (label_indices, label_names) = zip(*label_dict.items())

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
