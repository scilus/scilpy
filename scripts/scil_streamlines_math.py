#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performs an operation on a list of streamline files. The supported
operations are:

    difference:  Keep the streamlines from the first file that are not in
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

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.streamlines import (difference, intersection, union, sum_sft)


OPERATIONS = {
    'difference': difference,
    'intersection': intersection,
    'union': union,
    'concatenate': 'concatenate'
}


def _build_arg_parser():

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
    # TODO
    p.add_argument('--precision', '-p', metavar='NUMBER_OF_DECIMALS', type=int,
                   help='The precision used when comparing streamlines.')

    p.add_argument('--no_metadata', '-n', action='store_true',
                   help='Strip the streamline metadata from the output.')
    # TODO
    p.add_argument('--save_indices', '-s', metavar='OUTPUT_INDEX_FILE',
                   help='Save the streamline indices to the supplied '
                   'json file.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, args.inputs)
    assert_outputs_exist(parser, args, args.output)

    # Load all input streamlines.
    sft_list = [load_tractogram_with_reference(parser, args, f) for f in args.inputs]
    new_sft = sum_sft(sft_list, args.no_metadata)
    nb_streamlines = [len(sft) for sft in sft_list]

    # Apply the requested operation to each input file.
    logging.info(
        'Performing operation \'{}\'.'.format(args.operation))
    if args.operation == 'concatenate':
        indices = range(len(new_sft))
    else:
        _, indices = OPERATIONS[args.operation](new_sft.streamlines)

    # Save the indices to a file if requested.
    if args.save_indices is not None:
        start = 0
        indices_dict = {'filenames': args.inputs}
        for name, nb in zip(args.inputs, nb_streamlines):
            end = start + nb
            file_indices = \
                [i - start for i in indices if start <= i < end]
            indices_dict[name] = file_indices
            start = end
        with open(args.save_indices, 'wt') as f:
            json.dump(indices_dict, f)

    # Save the new streamlines.
    logging.info('Saving streamlines to {0}.'.format(args.output))
    save_tractogram(new_sft[indices], args.output)


if __name__ == "__main__":
    main()
