#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add, extract or delete information to each streamline from a tractogram
file. Can be for example SIFT2 weights, processing information, bundle IDs,
etc.

Input and output tractograms must always be .trk to simplify filetype checks
for each operation.
"""

import os
import argparse
import logging

from dipy.io.streamline import save_tractogram, load_tractogram
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             check_tract_trk,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format)


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Input tractogram (.trk).')
    p.add_argument('operation', metavar='OPERATION',
                   choices=['add', 'delete', 'export'],
                   help='The type of operation to be performed on the\n'
                        'tractogram\'s data_per_streamline. Must be one of\n'
                        'the following: [%(choices)s]. The arguments\n'
                        'required for each operation are specified under\n'
                        'each group below.')

    add_args = p.add_argument_group('Add operation',
                                    'Requires the out_tractogram argument.')
    add_args.add_argument('--add_dps_file',
                          help='File containing the data to add to\n'
                               'streamlines.')
    add_args.add_argument('--add_dps_key', type=str,
                          help='Where to store the data in the tractogram.')

    delete_args = p.add_argument_group('Delete operation',
                                       'Requires the out_tractogram argument.')
    delete_args.add_argument('--delete_dps_key', type=str,
                             help='Where to find the data to be deleted, in\n'
                                  'the tractogram.')

    p.add_argument('--out_tractogram',
                   help='Output tractogram (.trk). Required for any mutation.')

    export_args = p.add_argument_group('Export operation')
    export_args.add_argument('--export_dps_key', type=str,
                             help='Where to find the data to be exported,\n'
                                  'in the tractogram.')
    export_args.add_argument('--export_dps_file',
                             help='File in which the extracted data will be\n'
                                  'saved.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    check_tract_trk(parser, args.in_tractogram)
    if args.out_tractogram:
        check_tract_trk(parser, args.out_tractogram)

    # I/O assertions
    assert_inputs_exist(parser, args.in_tractogram, args.add_dps_file)
    assert_outputs_exist(parser, args, [], optional=[args.export_dps_file,
                                                     args.out_tractogram])

    sft = load_tractogram(args.in_tractogram, 'same')

    if args.operation == 'add':
        # Make sure the user is not unwillingly overwritting dps
        if (args.add_dps_key in sft.get_data_per_streamline_keys() and
                not args.overwrite):
            parser.error('"{}" already in data per streamline. Use -f to force'
                         ' overwriting.'.format(args.add_dps_key))

        # Load data and remove extraneous dimensions
        data = np.squeeze(load_matrix_in_any_format(args.add_dps_file))

        # Quick check as the built-in error from sft is not too explicit
        if len(sft) != data.shape[0]:
            raise ValueError('Data must have as many entries ({}) as there are'
                             ' streamlines ({}).'.format(data.shape[0],
                                                         len(sft)))

        sft.data_per_streamline[args.add_dps_key] = data

        save_tractogram(sft, args.out_tractogram)

    if args.operation == 'delete':
        del sft.data_per_streamline[args.delete_dps_key]

        save_tractogram(sft, args.out_tractogram)

    if args.operation == 'export':
        # Extract data and reshape
        if not args.export_dps_key in sft.data_per_streamline.keys():
            raise ValueError('Data does not have any data_per_streamline'
                             ' entry stored at this key: {}'
                                .format(args.export_dps_key))

        data = np.squeeze(sft.data_per_streamline[args.export_dps_key])
        save_matrix_in_any_format(args.export_dps_file, data)


if __name__ == '__main__':
    main()
