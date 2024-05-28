#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-

"""
Validate TRX file.

Removes streamlines that are out of the volume bounding box. In voxel space,
no negative coordinate and no above volume dimension coordinate are possible.
Any streamline that do not respect these two conditions are removed.

Also removes streamlines with single or no point.
The --remove_identical_streamlines option will remove identical streamlines.
'identical' is defined as having the same number of points and the same
points coordinates (to a specified precision, using a hash table).
"""

import argparse
import os

from trx.workflows import validate_tractogram


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy, trx.')
    p.add_argument('--out_tractogram',
                   help='Filename of the tractogram after removing invalid '
                        'streamlines.')
    p.add_argument('--remove_identical_streamlines', action='store_true',
                   help='Remove identical streamlines from the set.')
    p.add_argument('--precision', type=int, default=1,
                   help='Number of decimals to keep when hashing the points '
                        'of streamlines [%(default)s].')

    p.add_argument('--reference',
                   help='Reference anatomy for tck/vtk/fib/dpy file\n'
                        'support (.nii or .nii.gz).')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.out_tractogram and os.path.isfile(args.out_tractogram) \
            and not args.overwrite:
        raise IOError('{} already exists, use -f to overwrite.'.format(
            args.out_tractogram))

    validate_tractogram(args.in_tractogram, reference=args.reference,
                        out_tractogram=args.out_tractogram,
                        remove_identical_streamlines=args.remove_identical_streamlines,
                        precision=args.precision)


if __name__ == "__main__":
    main()
