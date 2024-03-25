#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversion of tensors (the 6 values from the triangular matrix) between various
software standards. We cannot discover the input format type, user must know
how the tensors were created.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.io.tensor import (supported_tensor_formats,
                              tensor_format_description,
                              convert_tensor_format)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__ + tensor_format_description,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_file',
                   help='Input tensors filename.')
    p.add_argument('out_file',
                   help='Output tensors filename.')
    p.add_argument('in_format', metavar='in_format',
                   choices=supported_tensor_formats,
                   help='Input format. Choices: {}'
                   .format(supported_tensor_formats))
    p.add_argument('out_format', metavar='out_format',
                   choices=supported_tensor_formats,
                   help='Output format. Choices: {}'
                   .format(supported_tensor_formats))
    
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_file)
    assert_outputs_exist(parser, args, args.out_file)

    in_tensors_img = nib.load(args.in_file)
    in_tensors = in_tensors_img.get_fdata(dtype=np.float32)

    out_tensors = convert_tensor_format(in_tensors, args.in_format,
                                        args.out_format)
    out_tensors_img = nib.Nifti1Image(
        out_tensors.astype(np.float32), in_tensors_img.affine)
    nib.save(out_tensors_img, args.out_file)


if __name__ == "__main__":
    main()
