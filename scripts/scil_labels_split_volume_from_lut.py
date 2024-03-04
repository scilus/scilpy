#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a label image into multiple images where the name of the output images
is taken from a lookup table (ex: left-lateral-occipital.nii.gz,
right-thalamus.nii.gz, ...). Only the labels included in the lookup table
are extracted.

IMPORTANT: your label image must be of an integer type.

Formerly: scil_split_volume_by_labels.py
"""

import argparse
import json
import logging
import os

import nibabel as nib

from scilpy.image.labels import get_data_as_labels, get_lut_dir, split_labels
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             assert_output_dirs_exist_and_empty)


def _build_arg_parser():
    luts = [os.path.splitext(f)[0] for f in os.listdir(get_lut_dir())]

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_label',
                   help='Path of the input label file, in a format supported '
                        'by Nibabel.')
    p.add_argument('--out_dir', default='',
                   help='Put all ouptput images in a specific directory.')
    p.add_argument('--out_prefix', default='',
                   help='Prefix to be used for each output image.')

    mutual_group = p.add_mutually_exclusive_group(required=True)
    mutual_group.add_argument(
        '--scilpy_lut', choices=luts,
        help='Lookup table, in the file scilpy/data/LUT, used to name the '
             'output files.')
    mutual_group.add_argument(
        '--custom_lut',
        help='Path of the lookup table file, used to name the output files.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_label)

    label_img = nib.load(args.in_label)
    label_img_data = get_data_as_labels(label_img)

    if args.scilpy_lut:
        with open(os.path.join(get_lut_dir(), args.scilpy_lut + '.json')) as f:
            label_dict = json.load(f)
    else:
        with open(args.custom_lut) as f:
            label_dict = json.load(f)

    output_filenames = []
    for label, name in label_dict.items():
        if int(label) != 0:
            if args.out_prefix:
                output_filenames.append(os.path.join(
                    args.out_dir,
                    '{0}_{1}.nii.gz'.format(args.out_prefix, name)))
            else:
                output_filenames.append(os.path.join(
                    args.out_dir, '{0}.nii.gz'.format(name)))

    assert_output_dirs_exist_and_empty(parser, args, [], optional=args.out_dir)
    assert_outputs_exist(parser, args, output_filenames)

    # Extract the voxels that match the label and save them to a file.
    label_indices = list(label_dict.keys())
    split_data = split_labels(label_img_data, label_indices)

    for i in range(len(label_indices)):
        if split_data[i] is not None:
            split_image = nib.Nifti1Image(split_data[i],
                                          label_img.affine,
                                          header=label_img.header)
            nib.save(split_image, output_filenames[i])


if __name__ == "__main__":
    main()
