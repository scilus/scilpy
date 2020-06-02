#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Print the raw header from the provided file or only the specified keys.
Support trk, nii and mgz files.
"""

import argparse
import pprint

import nibabel as nib

from scilpy.io.utils import assert_inputs_exist
from scilpy.utils.filenames import split_name_with_nii


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_file',
                   help='Input file (trk, nii and mgz).')
    p.add_argument('--keys', nargs='+',
                   help='Print only the specified keys.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_file)

    _, in_extension = split_name_with_nii(args.in_file)

    if in_extension in ['.tck', '.trk']:
        header = nib.streamlines.load(args.in_file, lazy_load=True).header
    elif in_extension in ['.nii', '.nii.gz', '.mgz']:
        header = dict(nib.load(args.in_file).header)
    else:
        parser.error('{} is not a supported extension.'.format(in_extension))

    if args.keys:
        for key in args.keys:
            if key not in header:
                parser.error('Key {} is not in the header of {}.'.format(key,
                             args.in_file))
            print(" '{}': {}".format(key, header[key]))
    else:
        pp = pprint.PrettyPrinter(indent=1)
        pp.pprint(header)


if __name__ == "__main__":
    main()
