#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converts a Varian FDF file or directory to a nifti file.
If the procpar contains diffusion information, it will be saved as bval and
bvec in the same folder as the output file.

ex: scil_dwi_convert_FDF.py semsdw/b0_folder/ semsdw/dwi_folder/ \
        dwi.nii.gz --bval dwi.bval --bvec dwi.bvec -f

Formerly: scil_convert_fdf.py
"""

import argparse
import logging

from scilpy.io.varian_fdf import (correct_procpar_intensity, load_fdf,
                                  save_babel)
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,)

    p.add_argument('in_b0_path',
                   help='Path to the b0 FDF file or folder to convert.')
    p.add_argument('in_dwi_path',
                   help='Path to the DWI FDF file or folder to convert.')
    p.add_argument('out_path',
                   help='Path to the nifti file to write on disk.')
    p.add_argument('--bval',
                   help='Path to the bval file to write on disk.')
    p.add_argument('--bvec',
                   help='Path to the bvec file to write on disk.')
    p.add_argument('--flip', metavar='dimension', default=None,
                   choices=['x', 'y', 'z'], nargs='+',
                   help='The axes you want to flip. eg: to flip the x '
                        'and y axes use: x y. [%(default)s]')
    p.add_argument('--swap', metavar='dimension', default=None,
                   choices=['x', 'y', 'z'], nargs='+',
                   help='The axes you want to swap. eg: to swap the x '
                        'and y axes use: x y. [%(default)s]')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_outputs_exist(parser, args, args.out_path, optional=[args.bval,
                                                                args.bvec])

    data_dwi, header_dwi = load_fdf(args.in_dwi_path)
    data_b0, header_b0 = load_fdf(args.in_b0_path)

    data_dwi = correct_procpar_intensity(data_dwi, args.in_dwi_path,
                                         args.in_b0_path)

    save_babel(data_dwi, header_dwi,
               data_b0, header_b0,
               args.bval, args.bvec,
               args.out_path,
               flip=args.flip,
               swap=args.swap)


if __name__ == "__main__":
    main()
