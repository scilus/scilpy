#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from dipy.io.utils import is_header_compatible

from scilpy.io.utils import assert_inputs_exist
from scilpy.utils.filenames import split_name_with_nii

DESCRIPTION = """
"""


def _build_args_parser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_files', nargs='+',
                   help='')

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_files)

    for filepath in args.in_files:
        _, in_extension = split_name_with_nii(filepath)
        if in_extension not in ['.trk', '.nii', '.nii.gz']:
            parser.error('{} does not have a supported extension'.format(
                filepath))
        if not is_header_compatible(args.in_files[0], filepath):
            print('{} and {} do not have compatible header.'.format(
                args.in_files[0], filepath))


if __name__ == "__main__":
    main()
