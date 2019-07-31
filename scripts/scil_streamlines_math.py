#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performs an operation on a list of streamline files. The supported
operations are:

    subtraction:  Keep the streamlines from the first file that are not in
                  any of the following files.

    intersection: Keep the streamlines that are present in all files.

    union:        Keep all streamlines while removing duplicates.

    concatenate:  Keep all streamlines with duplicates.

For efficiency, the comparisons are performed using a hash table. This means
that streamlines must be identical for a match to be found. To allow a soft
match, use the --precision option to round streamlines before processing.
Note that the streamlines that are saved in the output are the original
streamlines, not the rounded ones.

The metadata (data per point, data per streamline) of the streamlines that
are kept in the output will preserved. This requires that all input files
share the same type of metadata. If this is not the case, use the option
--no-data to strip the metadata from the output.

Repeated uses with .trk files will slighly affect coordinate values
due to precision error.
"""

import argparse
from itertools import chain
import json
import logging

from nibabel.streamlines import load, save, Tractogram
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exists)
from scilpy.utils.streamlines import (perform_streamlines_operation,
                                      subtraction, intersection, union)


OPERATIONS = {
    'subtraction': subtraction,
    'intersection': intersection,
    'union': union,
    'concatenate': 'concatenate'
}


def build_args_p():

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('operation', choices=OPERATIONS.keys(), metavar='OPERATION',
                   help='The type of operation to be performed on the '
                   'streamlines. Must\nbe one of the following: '
                   '%(choices)s.')

    p.add_argument('inputs', metavar='INPUT_FILES', nargs='+',
                   help='The list of files that contain the ' +
                   'streamlines to operate on.')

    p.add_argument('output', metavar='OUTPUT_FILE',
                   help='The file where the remaining streamlines '
                   'are saved.')

    p.add_argument('--precision', '-p', metavar='NUMBER_OF_DECIMALS', type=int,
                   help='The precision used when comparing streamlines.')

    p.add_argument('--no_metadata', '-n', action='store_true',
                   help='Strip the streamline metadata from the output.')

    p.add_argument('--save_metadata_indices', '-m', action='store_true',
                   help='Save streamline indices to metadata. Has no '
                   'effect if --no-data\nis present. Will '
                   'overwrite \'ids\' metadata if already present.')

    p.add_argument('--save_indices', '-s', metavar='OUTPUT_INDEX_FILE',
                   help='Save the streamline indices to the supplied '
                   'json file.')

    p.add_argument('--verbose', '-v', action='store_true', dest='verbose',
                   help='Produce verbose output.')

    add_overwrite_arg(p)

    return p


def load_data(path):
    logging.info(
        'Loading streamlines from {0}.'.format(path))
    tractogram = load(path).tractogram
    streamlines = list(tractogram.streamlines)
    data_per_streamline = tractogram.data_per_streamline
    data_per_point = tractogram.data_per_point

    return streamlines, data_per_streamline, data_per_point


def main():

    p = build_args_p()
    args = p.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(p, args.inputs)
    assert_outputs_exists(p, args, [args.output])

    # Load all input streamlines.
    data = [load_data(f) for f in args.inputs]
    streamlines, data_per_streamline, data_per_point = zip(*data)
    nb_streamlines = [len(s) for s in streamlines]

    # Apply the requested operation to each input file.
    logging.info(
        'Performing operation \'{}\'.'.format(args.operation))
    if args.operation == 'concatenate':
        new_streamlines = sum(streamlines, [])
        indices = range(len(new_streamlines))
    else:
        new_streamlines, indices = perform_streamlines_operation(
            OPERATIONS[args.operation], streamlines, args.precision)

    # Get the meta data of the streamlines.
    new_data_per_streamline = {}
    new_data_per_point = {}
    if not args.no_metadata:

        for key in data_per_streamline[0].keys():
            all_data = np.vstack([s[key] for s in data_per_streamline])
            new_data_per_streamline[key] = all_data[indices, :]

        # Add the indices to the metadata if requested.
        if args.save_metadata_indices:
            new_data_per_streamline['ids'] = indices

        for key in data_per_point[0].keys():
            all_data = list(chain(*[s[key] for s in data_per_point]))
            new_data_per_point[key] = [all_data[i] for i in indices]

    # Save the indices to a file if requested.
    if args.save_indices is not None:
        start = 0
        indices_dict = {'filenames': args.inputs}
        for name, nb in zip(args.inputs, nb_streamlines):
            end = start + nb
            file_indices = \
                [i - start for i in indices if i >= start and i < end]
            indices_dict[name] = file_indices
            start = end
        with open(args.save_indices, 'wt') as f:
            json.dump(indices_dict, f)

    # Save the new streamlines.
    logging.info('Saving streamlines to {0}.'.format(args.output))
    reference_file = load(args.inputs[0], lazy_load=True)
    new_tractogram = Tractogram(
        new_streamlines, data_per_streamline=new_data_per_streamline,
        data_per_point=new_data_per_point, affine_to_rasmm=np.eye(4))

    reference_file.header['nb_streamlines'] = len(new_streamlines)
    save(new_tractogram, args.output, header=reference_file.header)


if __name__ == "__main__":
    main()
