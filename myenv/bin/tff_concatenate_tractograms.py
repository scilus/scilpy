#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-

"""
Concatenate multiple tractograms into one.

If the data_per_point or data_per_streamline is not the same for all
tractograms, the data must be deleted first.
"""

import argparse
import os

from trx.io import load, save
from trx.trx_file_memmap import TrxFile, concatenate


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractograms', nargs='+',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy, trx.')
    p.add_argument('out_tractogram',
                   help='Filename of the concatenated tractogram.')

    p.add_argument('--delete_dpv', action='store_true',
                   help='Delete the dpv if it exists. '
                        'Required if not all input has the same metadata.')
    p.add_argument('--delete_dps', action='store_true',
                   help='Delete the dps if it exists. '
                        'Required if not all input has the same metadata.')
    p.add_argument('--delete_groups', action='store_true',
                   help='Delete the groups if it exists. '
                        'Required if not all input has the same metadata.')
    p.add_argument('--reference',
                   help='Reference anatomy for tck/vtk/fib/dpy file\n'
                        'support (.nii or .nii.gz).')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_tractogram) and not args.overwrite:
        raise IOError('{} already exists, use -f to overwrite.'.format(
            args.out_tractogram))

    trx_list = []
    has_group = False
    for filename in args.in_tractograms:
        tractogram_obj = load(filename, args.reference)

        if not isinstance(tractogram_obj, TrxFile):
            tractogram_obj = TrxFile.from_sft(tractogram_obj)
        elif len(tractogram_obj.groups):
            has_group = True
        trx_list.append(tractogram_obj)

    trx = concatenate(trx_list, delete_dpv=args.delete_dpv,
                      delete_dps=args.delete_dps,
                      delete_groups=args.delete_groups or not has_group,
                      check_space_attributes=True,
                      preallocation=False)
    save(trx, args.out_tractogram)


if __name__ == "__main__":
    main()
