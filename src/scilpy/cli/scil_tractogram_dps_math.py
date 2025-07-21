#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Import, extract or delete dps (data_per_streamline) information to a tractogram
file. Can be for example SIFT2 weights, processing information, bundle IDs,
tracking seeds, etc.

This script is not the same as the dps mode of scil_tractogram_dpp_math.py,
which performs operations on dpp (data_per_point) and saves the result as dps.
Instead this script performs operations directly on dps values.

Input and output tractograms must be .trk, unless you are using the 'import'
operation, in which case a .tck input tractogram is accepted.

Usage examples:
    > scil_tractogram_dps_math.py tractogram.trk import "bundle_ids"
        --in_dps_file my_bundle_ids.txt
    > scil_tractogram_dps_math.py tractogram.trk export "seeds"
        --out_dps_file seeds.npy
"""

import nibabel as nib
import argparse
import logging

from dipy.io.streamline import save_tractogram, load_tractogram
from scilpy.io.streamlines import load_tractogram_with_reference
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             check_tract_trk,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractogram',
                   help='Input tractogram (.trk for all operations,'
                        '.tck accepted for import).')
    p.add_argument('operation', metavar='OPERATION',
                   choices=['import', 'delete', 'export'],
                   help='The type of operation to be performed on the\n'
                        'tractogram\'s data_per_streamline at the given\n'
                        'key. Must be one of the following: [%(choices)s].\n'
                        'The additional arguments required for each\n'
                        'operation are specified under each group below.')
    p.add_argument('dps_key', type=str,
                   help='Key name used for the operation.')

    p.add_argument('--out_tractogram',
                   help='Output tractogram (.trk). Required for "import" and\n'
                        '"delete" operations.')

    import_args = p.add_argument_group('Operation "import" mandatory options')
    import_excl = import_args.add_mutually_exclusive_group()
    import_excl.add_argument('--in_dps_file',
                             help='File containing the data to import to\n'
                                  'streamlines (.txt, .npy or .mat). There\n'
                                  'must be the same amount of entries as\n'
                                  'there are streamlines.')
    import_excl.add_argument('--in_dps_single_value', nargs='+', type=float,
                             help='Single value to import to each\n'
                                  'streamline. If the value is an array,\n'
                                  'enter each component with a space in\n'
                                  'between.')

    export_args = p.add_argument_group('Operation "export" mandatory options')
    export_args.add_argument('--out_dps_file',
                             help='File in which the extracted data will be\n'
                                  'saved (.txt or .npy).')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if args.operation == 'import':
        if not nib.streamlines.is_supported(args.in_tractogram):
            parser.error('Invalid input streamline file format (must be trk ' +
                         'or tck): {0}'.format(args.in_tractogram))
    else:
        check_tract_trk(parser, args.in_tractogram)

    if args.out_tractogram:
        check_tract_trk(parser, args.out_tractogram)

    assert_inputs_exist(parser, args.in_tractogram, args.in_dps_file)
    assert_outputs_exist(parser, args, [], optional=[args.out_dps_file,
                                                     args.out_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    if args.operation == 'import':
        if args.in_dps_file is None and args.in_dps_single_value is None:
            parser.error('One of --in_dps_file or ' +
                         '--in_dps_single_value is required for the ' +
                         '"import" operation.')

        if args.out_tractogram is None:
            parser.error('The --out_tractogram option is required for ' +
                         'the "import" operation.')

        # Make sure the user is not unwillingly overwritting dps
        if (args.dps_key in sft.get_data_per_streamline_keys() and
                not args.overwrite):
            parser.error('"{}" already in data per streamline. Use -f to force'
                         ' overwriting.'.format(args.dps_key))

        if args.in_dps_file:
            # Load data and remove extraneous dimensions
            data = np.squeeze(load_matrix_in_any_format(args.in_dps_file))

            # Validate data shape
            if len(sft) != data.shape[0]:
                raise ValueError(
                    'Data must have as many entries ({}) as there are '
                    'streamlines ({}).'.format(data.shape[0], len(sft)))
        elif args.in_dps_single_value:
            data = np.array(args.in_dps_single_value)
            if (np.mod(data, 1) == 0).all():
                data = data.astype(int)

            # Squeeze may remove axes of length 0, but still returns an
            # ndarray. We would like a proper scalar type.
            if len(data) == 1:
                data = data[0]

            data = [data] * len(sft.streamlines)

        sft.data_per_streamline[args.dps_key] = data

        save_tractogram(sft, args.out_tractogram)

    if args.operation == 'delete':
        if args.out_tractogram is None:
            parser.error('The --out_tractogram option is required for ' +
                         'the "delete" operation.')

        del sft.data_per_streamline[args.dps_key]

        save_tractogram(sft, args.out_tractogram)

    if args.operation == 'export':
        if args.out_dps_file is None:
            parser.error('The --out_dps_file option is required for ' +
                         'the "export" operation.')

        # Extract data and reshape
        if args.dps_key not in sft.data_per_streamline.keys():
            raise ValueError('Data does not have any data_per_streamline'
                             ' entry stored at this key: {}'
                             .format(args.dps_key))

        data = np.squeeze(sft.data_per_streamline[args.dps_key])
        save_matrix_in_any_format(args.out_dps_file, data)


if __name__ == '__main__':
    main()
