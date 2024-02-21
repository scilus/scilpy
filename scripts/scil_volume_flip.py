#! /usr/bin/env python3

"""
Flip the volume according to the specified axis.

Formerly: scil_flip_volume.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.image.volume_operations import flip_volume
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             add_verbose_arg, assert_outputs_exist)


def _build_arg_parser():

    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('in_image',
                   help='Path of the input volume (nifti).')
    p.add_argument('out_image',
                   help='Path of the output volume (nifti).')
    p.add_argument('axes', metavar='dimension',
                   choices=['x', 'y', 'z'], nargs='+',
                   help='The axes you want to flip. eg: to flip the x '
                        'and y axes use: x y.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image)
    assert_outputs_exist(parser, args, args.out_image)

    vol = nib.load(args.in_image)
    data = vol.get_fdata(dtype=np.float32)
    affine = vol.affine
    header = vol.header

    data = flip_volume(data, args.axes)

    nib.save(nib.Nifti1Image(data, affine, header=header), args.out_image)


if __name__ == "__main__":
    main()
