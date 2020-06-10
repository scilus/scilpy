#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Will compare all input files against the first one for the compatibility
of their spatial attributes.

Spatial attributes are: affine, dimensions, voxel sizes and voxel order.
"""

import argparse

from dipy.io.utils import is_header_compatible

from scilpy.io.utils import assert_inputs_exist
from scilpy.utils.filenames import split_name_with_nii


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_files', nargs='+',
                   help='List of file to compare (trk and nii).')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_files)

    all_valid = True
    for filepath in args.in_files:
        _, in_extension = split_name_with_nii(filepath)
        if in_extension not in ['.trk', '.nii', '.nii.gz']:
            parser.error('{} does not have a supported extension'.format(
                filepath))
        if not is_header_compatible(args.in_files[0], filepath):
            print('{} and {} do not have compatible header.'.format(
                args.in_files[0], filepath))
            all_valid = False
    if all_valid:
        print('All input files have compatible headers.')


if __name__ == "__main__":
    main()
