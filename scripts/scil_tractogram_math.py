#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs an operation on a list of streamline files. The supported
operations are:

difference:  Keep the streamlines from the first file that are not in
                any of the following files.

intersection: Keep the streamlines that are present in all files.

union:        Keep all streamlines while removing duplicates.

concatenate:  Keep all streamlines with duplicates.

lazy_concatenate:  Keep all streamlines with duplicates, never load the whole
                    tractograms in memory. Only works with trk/tck file,
                    metadata will be lost and invalid streamlines are kept.

If a file 'duplicate.trk' have identical streamlines, calling the script using
the difference/intersection/union with a single input will remove these
duplicated streamlines.

To allow a soft match, use the --precision option to increase the allowed
threshold for similarity. A precision of 1 represents 10**(-1), so a
maximum distance of 0.1mm is allowed. If the streamlines are identical, the
default value of 3 (or 0.001mm distance) should work.

If there is a 0.5mm shift, use a precision of 0 (or 1mm distance) and the
--robust option. Should make it work, but slightly slower. Will merge all
streamlines similar when rounded to that precision level.

The metadata (data per point, data per streamline) of the streamlines that
are kept in the output will be preserved. This requires that all input files
share the same type of metadata. If this is not the case, use the option
--no_metadata to strip the metadata from the output. Or --fake_metadata to
initialize dummy metadata in the file missing them.

Formerly: scil_streamlines_math.py
"""

import argparse
import json
import logging
import os

from dipy.io.streamline import save_tractogram
import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_bbox_arg,
                             add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_headers_compatible)
from scilpy.tractograms.lazy_tractogram_operations import lazy_concatenate
from scilpy.tractograms.tractogram_operations import (
    perform_tractogram_operation_on_sft, concatenate_sft)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('operation', metavar='OPERATION',
                   choices=['difference', 'intersection', 'union',
                            'concatenate', 'lazy_concatenate'],
                   help='The type of operation to be performed on the '
                        'streamlines. Must\nbe one of the following: '
                        '%(choices)s.')
    p.add_argument('in_tractograms', metavar='INPUT_FILES', nargs='+',
                   help='The list of files that contain the ' +
                        'streamlines to operate on.')
    p.add_argument('out_tractogram', metavar='OUTPUT_FILE',
                   help='The file where the remaining streamlines '
                        'are saved.')

    p.add_argument('--precision', '-p', metavar='NBR_OF_DECIMALS',
                   type=int, default=4,
                   help='Precision used to compare streamlines [%(default)s].')
    p.add_argument('--robust', '-r', action='store_true',
                   help='Use version robust to small translation/rotation.')

    p.add_argument('--no_metadata', '-n', action='store_true',
                   help='Strip the streamline metadata from the output.')
    p.add_argument('--fake_metadata', action='store_true',
                   help='Skip the metadata verification, create fake metadata '
                        'if missing, can lead to unexpected behavior.')
    p.add_argument('--save_indices', '-s', metavar='OUT_INDEX_FILE',
                   help='Save the streamline indices to the supplied '
                        'json file.')
    p.add_argument('--save_empty', action='store_true',
                   help="If set, we will save all results, even if tractogram "
                        "if empty.")

    add_bbox_arg(p)
    add_json_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractograms, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram,
                         optional=args.save_indices)
    assert_headers_compatible(parser, args.in_tractograms,
                              reference=args.reference)

    if args.operation == 'lazy_concatenate':
        logging.info('Using lazy_concatenate, no spatial or metadata related '
                     'checks are performed.\nMetadata will be lost, only '
                     'trk/tck file are supported.\n To use trk, at least one '
                     'input must be a trk.')
        _, out_ext = os.path.splitext(args.out_tractogram)

        # In some cases, if -f is used and previous file contained errors
        # (ex, wrong header), the lazy version does not overwrite the file
        # completely. Deleting manually
        if os.path.isfile(args.out_tractogram) and args.overwrite:
            os.remove(args.out_tractogram)

        out_tractogram, header = lazy_concatenate(args.in_tractograms, out_ext)
        nib.streamlines.save(out_tractogram, args.out_tractogram,
                             header=header)
        return

    # Load all input streamlines.
    sft_list = []
    for f in args.in_tractograms:
        logging.info("Loading file {}".format(f))
        # Using in a millimeter space so that the precision level is in mm.
        # Note. Sending to_voxmm() returns None with no streamlines.
        tmp_sft = load_tractogram_with_reference(parser, args, f)
        tmp_sft.to_voxmm()

        sft_list.append(tmp_sft)

    if np.all([len(sft) == 0 for sft in sft_list]):
        return

    # Apply the requested operation to each input file.
    if args.operation == 'concatenate':
        logging.info('Performing operation "concatenate"')
        sft_list = [s for s in sft_list if s is not None]
        new_sft = concatenate_sft(sft_list, args.no_metadata,
                                  args.fake_metadata)
        indices_per_sft = [np.arange(len(new_sft), dtype=np.uint32)]
    else:
        op_name = args.operation
        if args.robust:
            op_name += '_robust'

        logging.info('Performing operation \'{}\'.'.format(op_name))
        new_sft, indices_per_sft = perform_tractogram_operation_on_sft(
            op_name, sft_list, precision=args.precision,
            no_metadata=args.no_metadata, fake_metadata=args.fake_metadata)

        if len(new_sft) == 0 and not args.save_empty:
            logging.info("Empty resulting tractogram. Not saving results.")
            return

    # Save the indices to a file if requested.
    if args.save_indices:
        out_dict = {}
        for name, ind in zip(args.in_tractograms, indices_per_sft):
            # Switch to int32 for json
            out_dict[name] = ind

        with open(args.save_indices, 'wt') as f:
            json.dump(out_dict, f, indent=args.indent,
                      sort_keys=args.sort_keys)

    # Save the new streamlines (and metadata)
    logging.info('Saving {} streamlines to {}.'.format(len(new_sft),
                                                       args.out_tractogram))
    save_tractogram(new_sft, args.out_tractogram,
                    bbox_valid_check=args.bbox_check)


if __name__ == "__main__":
    main()
