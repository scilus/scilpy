#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Add information to each streamline from a file. Can be for example
SIFT2 weights, processing information, bundle IDs, etc.

Output must be a .trk otherwise the data will be lost.
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             check_tract_trk,
                             load_matrix_in_any_format)


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Input tractogram (.trk or .tck).')
    p.add_argument('in_dps_file',
                   help='File containing the data to add to streamlines.')
    p.add_argument('dps_key',
                   help='Where to store the data in the tractogram.')
    p.add_argument('out_tractogram',
                   help='Output tractogram (.trk).')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # I/O assertions
    assert_inputs_exist(parser, [args.in_tractogram, args.in_dps_file],
                        args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram)
    check_tract_trk(parser, args.out_tractogram)

    # Load tractogram
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    # Make sure the user is not unwillingly overwritting dps
    if (args.dps_key in sft.get_data_per_streamline_keys() and
       not args.overwrite):
        parser.error('"{}" already in data per streamline. Use -f to force '
                     'overwriting.'.format(args.dps_key))

    # Load data and remove extraneous dimensions
    data = np.squeeze(load_matrix_in_any_format(args.in_dps_file))

    # Quick check as the built-in error from sft is not too explicit
    if len(sft) != data.shape[0]:
        raise ValueError('Data must have as many entries ({}) as there are'
                         ' streamlines ({}).'.format(data.shape[0], len(sft)))
    # Add data to tractogram
    sft.data_per_streamline[args.dps_key] = data

    # Save the new sft
    save_tractogram(sft, args.out_tractogram)


if __name__ == '__main__':
    main()
