#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pprint

import nibabel as nib

from scilpy.io.utils import assert_inputs_exist
from scilpy.utils.filenames import split_name_with_nii

DESCRIPTION = """
"""


def _build_args_parser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_file',
                   help='')
    p.add_argument('--keys', nargs='+',
                   help='')

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_file)

    _, in_extension = split_name_with_nii(args.in_file)

    if in_extension in ['.tck', '.trk']:
        header = nib.streamlines.load(args.in_file, lazy_load=True).header
    elif in_extension in ['.nii', '.nii.gz', '.mgz']:
        header = dict(nib.load(args.in_file).header)
    else:
        parser.error('{} is not a supported extension.', in_extension)

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
