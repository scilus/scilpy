#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Will compare all input files against the first one for the compatibility
of their spatial attributes.

Spatial attributes are: affine, dimensions, voxel sizes and voxel order.

Formerly: scil_verify_space_attributes_compatibility.py
"""

import argparse
import logging

from scilpy.io.utils import (
    add_reference_arg,
    add_verbose_arg,
    assert_inputs_exist,
    assert_headers_compatible)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_files', nargs='+',
                   help='List of file to compare (trk, tck and nii/nii.gz).')

    add_reference_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_files, args.reference)
    assert_headers_compatible(parser, args.in_files,
                              reference=args.reference)

    # If we come here, it means there was no error.
    print('All input files have compatible headers.')


if __name__ == "__main__":
    main()
