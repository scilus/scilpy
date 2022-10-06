#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Will compare all input files against the first one for the compatibility
of their spatial attributes.

Spatial attributes are: affine, dimensions, voxel sizes and voxel order.
"""

import argparse


from scilpy.io.utils import assert_inputs_exist, is_header_compatible_multiple_files


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
    is_header_compatible_multiple_files(parser, args.in_files,
                                        verbose_all_compatible=True)


if __name__ == "__main__":
    main()
