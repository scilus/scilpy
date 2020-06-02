#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reshape / reslice / resample *.nii or *.nii.gz using a reference.
For more information on how to use the various registration scripts
see the doc/tractogram_registration.md readme file.

>>> scil_reshape_to_reference.py wmparc.mgz t1.nii.gz wmparc_t1.nii.gz \\
    --interpolation nearest
"""

import argparse

import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.image import transform_anatomy


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_file',
                   help='Path of the volume file to be reshaped.')

    p.add_argument('ref_file',
                   help='Path of the reference volume.')

    p.add_argument('out_file',
                   help='Output filename of the reshaped data.')

    p.add_argument('--interpolation', default='linear',
                   choices=['linear', 'nearest'],
                   help='Interpolation: "linear" or "nearest". [%(default)s]')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_file, args.ref_file])
    assert_outputs_exist(parser, args, args.out_file)

    transform_anatomy(np.eye(4), args.ref_file, args.in_file,
                      args.out_file, interp=args.interpolation)


if __name__ == "__main__":
    main()
