#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a label image into multiple images where the name of the output images
is taken from a lookup table (ex: left-lateral-occipital.nii.gz,
right-thalamus.nii.gz, ...). Only the labels included in the lookup table
are extracted.

IMPORTANT: your label image must be of an integer type.
"""

import argparse
import inspect
import json
import os

import nibabel as nib
import numpy as np

import scilpy
from scilpy.io.image import get_data_as_label
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    luts = [os.path.splitext(f)[0] for f in os.listdir(get_lut_dir())]

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_label',
                   help='Path of the input label file, '
                        'in a format supported by Nibabel.')
    p.add_argument('--out_dir', default='',
                   help='Put all ouptput images in a specific directory.')
    p.add_argument('--out_prefix', default='',
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


def get_lut_dir():
    """
    Return LUT directory in scilpy repository

    Returns
    -------
    lut_dir: string
        LUT path
    """
    # Get the valid LUT choices.
    module_path = inspect.getfile(scilpy)

    lut_dir = os.path.join(os.path.dirname(
        os.path.dirname(module_path)) + "/data/LUT/")

    return lut_dir


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    required = args.in_label
    assert_inputs_exist(parser, required)

    label_img = nib.load(args.in_label)
    label_img_data = get_data_as_label(label_img)

    if args.scilpy_lut:
        with open(os.path.join(get_lut_dir(), args.scilpy_lut + '.json')) as f:
            label_dict = json.load(f)
        (label_indices, label_names) = zip(*label_dict.items())
    else:
        with open(args.custom_lut) as f:
            label_dict = json.load(f)
        (label_indices, label_names) = zip(*label_dict.items())

    output_filenames = []
    for label, name in zip(label_indices, label_names):
        if int(label) != 0:
            if args.out_prefix:
                output_filenames.append(os.path.join(args.out_dir,
                                                     '{0}_{1}.nii.gz'.format(
                                                         args.out_prefix,
                                                         name)))
            else:
                output_filenames.append(os.path.join(args.out_dir,
                                                     '{0}.nii.gz'.format(
                                                        name)))

    assert_outputs_exist(parser, args, output_filenames)

    if args.out_dir and not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # Extract the voxels that match the label and save them to a file.
    cnt_filename = 0
    for label in label_indices:
        if int(label) != 0:
            split_label = np.zeros(label_img.shape,
                                   dtype=np.uint16)
            split_label[np.where(label_img_data == int(label))] = label

            split_image = nib.Nifti1Image(split_label,
                                          label_img.affine,
                                          header=label_img.header)
            nib.save(split_image, output_filenames[cnt_filename])
            cnt_filename += 1


if __name__ == "__main__":
    main()
