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

If there is a 0.5mm shift, use a precision of 0 (or 1mm distance) the --robust
option should make it work, but slightly slower.

The metadata (data per point, data per streamline) of the streamlines that
are kept in the output will preserved. This requires that all input files
share the same type of metadata. If this is not the case, use the option
--no_metadata to strip the metadata from the output. Or --fake_metadata to
initialize dummy metadata in the file missing them.
"""

import argparse
import json
import logging
import os

from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible
import nibabel as nib
from nibabel.streamlines import LazyTractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.streamlines import (difference_robust, difference,
                                      union_robust, union,
                                      intersection_robust, intersection,
                                      perform_streamlines_operation,
                                      concatenate_sft)


OPERATIONS = {
    'difference_robust': difference_robust,
    'intersection_robust': intersection_robust,
    'union_robust': union_robust,
    'difference': difference,
    'intersection': intersection,
    'union': union,
    'concatenate': 'concatenate',
    'lazy_concatenate': 'lazy_concatenate'
}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('operation', choices=OPERATIONS.keys(), metavar='OPERATION',
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

    p.add_argument('--ignore_invalid', action='store_true',
                   help='If set, does not crash because of invalid '
                        'streamlines.')

    add_json_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, args.in_tractograms)
    assert_outputs_exist(parser, args, args.out_tractogram,
                         optional=args.save_indices)

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

        def list_generator_from_nib(filenames):
            for in_file in filenames:
                logging.info("Lazy-loading file {}".format(in_file))
                tractogram_file = nib.streamlines.load(in_file, lazy_load=True)
                for s in tractogram_file.streamlines:
                    yield s

        # Verifying headers
        # Header will stay None for tck output. Will become a trk header (for
        # trk output) if we find at least one trk input.
        header = None
        for in_file in args.in_tractograms:
            _, ext = os.path.splitext(in_file)
            if ext == '.trk' and out_ext == '.trk':
                if header is None:
                    header = nib.streamlines.load(
                        in_file, lazy_load=True).header
                elif not is_header_compatible(header, in_file):
                    logging.warning('Incompatible headers in the list.')

        if out_ext == '.trk' and header is None:
            raise ValueError("No trk file encountered in the input list. "
                             "Result cannot be saved as a .trk.")

        generator = list_generator_from_nib(args.in_tractograms)
        out_tractogram = LazyTractogram(lambda: generator,
                                        affine_to_rasmm=np.eye(4))
        nib.streamlines.save(out_tractogram, args.out_tractogram,
                             header=header)
        return

    # Load all input streamlines.
    sft_list = []
    for f in args.in_tractograms:
        logging.info("Loading file {}".format(f))
        sft_list.append(load_tractogram_with_reference(
            parser, args, f, bbox_check=not args.ignore_invalid))

    # Apply the requested operation to each input file.
    logging.info('Performing operation \'{}\'.'.format(args.operation))
    new_sft = concatenate_sft(sft_list, args.no_metadata, args.fake_metadata)
    if args.operation == 'concatenate':
        indices = np.arange(len(new_sft), dtype=np.uint32)
    else:
        streamlines_list = [sft.streamlines for sft in sft_list]
        op_name = args.operation
        if args.robust:
            op_name += '_robust'
            _, indices = OPERATIONS[op_name](streamlines_list,
                                             precision=args.precision)
        else:
            _, indices = perform_streamlines_operation(
                OPERATIONS[op_name], streamlines_list,
                precision=args.precision)

    # Save the indices to a file if requested.
    if args.save_indices:
        start = 0
        out_dict = {}
        streamlines_len_cumsum = [len(sft) for sft in sft_list]
        for name, nb in zip(args.in_tractograms, streamlines_len_cumsum):
            end = start + nb
            # Switch to int32 for json
            out_dict[name] = [int(i - start)
                              for i in indices if start <= i < end]
            start = end

        with open(args.save_indices, 'wt') as f:
            json.dump(out_dict, f,
                      indent=args.indent,
                      sort_keys=args.sort_keys)

    # Save the new streamlines (and metadata)
    logging.info('Saving {} streamlines to {}.'.format(len(indices),
                                                       args.out_tractogram))
    save_tractogram(new_sft[indices], args.out_tractogram,
                    bbox_valid_check=not args.ignore_invalid)


if __name__ == "__main__":
    main()
