#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

"""
   Converts a Varian FDF file or directory to a nifti file.
   If the procpar contains diffusion information, it will be saved as bval and
   bvec in the same folder as the output file.
"""

from scilpy.io.varian_fdf import load_fdf, save_babel
from scilpy.io.utils import (add_overwrite_arg,
                             assert_outputs_exist)


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_path',
                   help='Path to the FDF file or folder to convert.')
    p.add_argument('out_path',
                   help='Path to the nifti file to write on disk.')
    p.add_argument('--bval',
                   help='Path to the bval file to write on disk.')
    p.add_argument('--bvec',
                   help='Path to the bvec file to write on disk.')
    add_overwrite_arg(p)
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    assert_outputs_exist(parser, args, args.out_path, optional=[args.bval,
                                                                args.bvec])

    data, header = load_fdf(args.in_path)
    save_babel(args.out_path, data, header, args.bval, args.bvec)


if __name__ == "__main__":
    main()
