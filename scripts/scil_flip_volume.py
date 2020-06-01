#! /usr/bin/env python3

"""
Flip the volume according to the specified axis.
"""

import argparse

import nibabel as nib

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():

    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('input',
                   help='Path of the input volume (nifti).')
    p.add_argument('output',
                   help='Path of the output volume (nifti).')
    p.add_argument('axes', metavar='dimension',
                   choices=['x', 'y', 'z'], nargs='+',
                   help='The axes you want to flip. eg: to flip the x '
                        'and y axes use: x y.')
    add_overwrite_arg(p)
    return p


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.input)
    assert_outputs_exist(parser, args, args.output)

    vol = nib.load(args.input)
    data = vol.get_data()
    affine = vol.get_affine()
    header = vol.get_header()

    if 'x' in args.axes:
        data = data[::-1, ...]

    if 'y' in args.axes:
        data = data[:, ::-1, ...]

    if 'z' in args.axes:
        data = data[:, :, ::-1, ...]

    nib.save(nib.Nifti1Image(data, affine, header), args.output)


if __name__ == "__main__":
    main()
