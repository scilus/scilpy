#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-

"""
This script is made to fix DSI-Studio TRK file (unknown space/convention) to
make it compatible with TrackVis, MI-Brain, Dipy Horizon (Stateful Tractogram).

The script either make it match with an anatomy from DSI-Studio.

This script was tested on various datasets and worked on all of them. However,
always verify the results and if a specific case does not work. Open an issue
on the Scilpy GitHub repository.

WARNING: This script is still experimental, DSI-Studio evolves quickly and
results may vary depending on the data itself as well as DSI-studio version.
"""

import argparse
import os

from trx.workflows import convert_dsi_studio


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dsi_tractogram', metavar='IN_DSI_TRACTOGRAM',
                   help='Path of the input tractogram file from DSI studio '
                        '(.trk).')
    p.add_argument('in_dsi_fa', metavar='IN_DSI_FA',
                   help='Path of the input FA from DSI Studio (.nii.gz).')
    p.add_argument('out_tractogram', metavar='OUT_TRACTOGRAM',
                   help='Path of the output tractogram file.')

    invalid = p.add_mutually_exclusive_group()
    invalid.add_argument('--remove_invalid', action='store_true',
                         help='Remove the streamlines landing out of the '
                              'bounding box.')
    invalid.add_argument('--keep_invalid', action='store_true',
                         help='Keep the streamlines landing out of the '
                              'bounding box.')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_tractogram) and not args.overwrite:
        raise IOError('{} already exists, use -f to overwrite.'.format(
            args.out_tractogram))

    convert_dsi_studio(args.in_dsi_tractogram, args.in_dsi_fa,
                       args.out_tractogram, remove_invalid=args.remove_invalid,
                       keep_invalid=args.keep_invalid)


if __name__ == "__main__":
    main()
