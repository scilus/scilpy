#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10

""" Simple comparison of tractogram by subtracting the coordinates' data.
Does not account for shuffling of streamlines. Simple A-B operations.

Differences below 1e^3 are expected for affine with large rotation/scaling.
Difference below 1e^6 are expected for isotropic data with small rotation.
"""

import argparse

from trx.workflows import tractogram_simple_compare


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractograms', nargs=2, metavar='IN_TRACTOGRAM',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy, trx.')
    p.add_argument('--reference', metavar='REFERENCE',
                   help='Reference anatomy for tck/vtk/fib/dpy file\n'
                   'support (.nii or .nii.gz).')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    tractogram_simple_compare(args.in_tractograms, args.reference)


if __name__ == "__main__":
    main()
