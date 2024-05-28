#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-

"""
Will compare all input files against the first one for the compatibility
of their spatial attributes.

Spatial attributes are: affine, dimensions, voxel sizes and voxel order.
"""

import argparse
from trx.workflows import verify_header_compatibility


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_files', nargs='+',
                   help='List of file to compare (trk, trx and nii).')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    verify_header_compatibility(args.in_files)


if __name__ == "__main__":
    main()
