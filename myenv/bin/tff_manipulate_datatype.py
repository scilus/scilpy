#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-

"""
Manipulate a TRX file internal array to change their data type.

Each instance of --dps, --dpv, --groups require 2 arguments (FILE, DTYPE).
--dpg requires 3 arguments (GROUP, FILE, DTYPE).
The choice of DTYPE are:
    - (u)int8, (u)int16, (u)int32, (u)int64
    - float16, float32, float64
    - bool

Example command:
tff_manipulate_datatype.py input.trx output.trx \
    --position float16 --offsets uint64 \
    --dpv color_x uint8 --dpv color_y uint8 --dpv color_z uint8 \
    --dpv fa float16 --dps algo uint8 --dps clusters_QB uint16 \
    --dps commit_colors uint8 --dps commit_weights float16 \
    --group CC uint64 --dpg CC mean_fa float64
"""

import argparse
import os

import numpy as np
from trx.workflows import manipulate_trx_datatype


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram',
                   help='Input TRX file.')
    p.add_argument('out_tractogram',
                   help='Output filename. Format must be one of\n'
                        'trk, tck, vtk, fib, dpy, trx.')

    p2 = p.add_argument_group(title='Data type options')
    p2.add_argument('--positions_dtype',
                    choices=['float16', 'float32', 'float64'],
                    help='Specify the datatype for positions for trx. '
                         '[%(choices)s]')
    p2.add_argument('--offsets_dtype',
                    choices=['uint32', 'uint64'],
                    help='Specify the datatype for offsets for trx. '
                         '[%(choices)s]')

    p3 = p.add_argument_group(title='Streamlines metadata options')
    p3.add_argument('--dpv', metavar=('NAME', 'DTYPE'), nargs=2,
                    action='append',
                    help='Specify the datatype for a specific data_per_vertex.')
    p3.add_argument('--dps', metavar=('NAME', 'DTYPE'), nargs=2,
                    action='append',
                    help='Specify the datatype for a specific data_per_streamline.')
    p3.add_argument('--groups', metavar=('NAME', 'DTYPE'), nargs=2,
                    action='append',
                    help='Specify the datatype for a specific group.')
    p3.add_argument('--dpg', metavar=('GROUP', 'NAME', 'DTYPE'), nargs=3,
                    action='append',
                    help='Specify the datatype for a specific data_per_group.')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_tractogram) and not args.overwrite:
        raise IOError('{} already exists, use -f to overwrite.'.format(
            args.out_tractogram))

    dtype_dict = {}
    if args.positions_dtype:
        dtype_dict['positions'] = np.dtype(args.positions_dtype)
    if args.offsets_dtype:
        dtype_dict['offsets'] = np.dtype(args.offsets_dtype)
    if args.dpv:
        dtype_dict['dpv'] = {}
        for name, dtype in args.dpv:
            dtype_dict['dpv'][name] = np.dtype(dtype)
    if args.dps:
        dtype_dict['dps'] = {}
        for name, dtype in args.dps:
            dtype_dict['dps'][name] = np.dtype(dtype)
    if args.groups:
        dtype_dict['groups'] = {}
        for name, dtype in args.groups:
            dtype_dict['groups'][name] = np.dtype(dtype)
    if args.dpg:
        dtype_dict['dpg'] = {}
        for group, name, dtype in args.dpg:
            dtype_dict['dpg'][group] = {name: np.dtype(dtype)}

    manipulate_trx_datatype(
        args.in_tractogram, args.out_tractogram, dtype_dict)


if __name__ == "__main__":
    main()
