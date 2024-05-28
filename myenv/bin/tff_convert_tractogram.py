#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-

"""
Conversion of '.tck', '.trk', '.fib', '.vtk', '.trx' and 'dpy' files using
updated file format standard. TCK file always needs a reference file, a NIFTI,
for conversion. The FIB file format is in fact a VTK, MITK Diffusion supports
it.
"""

import argparse
import os

from trx.workflows import convert_tractogram


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram', metavar='IN_TRACTOGRAM',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy, trx.')
    p.add_argument('out_tractogram', metavar='OUT_TRACTOGRAM',
                   help='Output filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy, trx.')

    p.add_argument('--reference',
                   help='Reference anatomy for tck/vtk/fib/dpy file\n'
                   'support (.nii or .nii.gz).')

    p2 = p.add_argument_group(title='Data type options')
    p2.add_argument('--positions_dtype', default='float32',
                    choices=['float16', 'float32', 'float64'],
                    help='Specify the datatype for positions for trx. [%(default)s]')
    p2.add_argument('--offsets_dtype', default='uint64',
                    choices=['uint32', 'uint64'],
                    help='Specify the datatype for offsets for trx. [%(default)s]')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_tractogram) and not args.overwrite:
        raise IOError('{} already exists, use -f to overwrite.'.format(
            args.out_tractogram))

    convert_tractogram(args.in_tractogram, args.out_tractogram, args.reference,
                       pos_dtype=args.positions_dtype,
                       offsets_dtype=args.offsets_dtype)


if __name__ == "__main__":
    main()
