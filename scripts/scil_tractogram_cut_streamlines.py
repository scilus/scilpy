#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filters streamlines and only keeps the parts of streamlines within (--mask) or
between (--label) the ROIs:

--mask

Streamlines outside of the mask will be cut. The mask may be disjoint. More
streamlines than input may be output if the mask is disjoint, therefore
data_per_streamline will be discared.

--label:

The script will cut streamlines so their longest segment is going from label 1
to label 2. Will keep data_per_streamline.

Both scenarios will erase data_per_point. Streamlines will be extended so they
reach the boundary of the mask or the two labels, therefore won't be equal
to the input streamlines.

Formerly: scil_cut_streamlines.py
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import compress_streamlines
import nibabel as nib

from scilpy.image.labels import get_data_as_labels
from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, assert_headers_compatible,
                             add_compression_arg)
from scilpy.tractograms.streamline_and_mask_operations import \
    cut_outside_of_mask_streamlines, cut_between_mask_two_blobs_streamlines
from scilpy.tractograms.streamline_operations import \
    resample_streamlines_step_size


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram',
                   help='Input tractogram file.')

    g1 = p.add_argument_group('Mandatory mask options',
                              'Choose between mask or label input.')
    g2 = g1.add_mutually_exclusive_group(required=True)
    g2.add_argument('--mask',
                    help='Binary mask.')
    g2.add_argument('--label',
                    help='Label containing 2 blobs.')

    p.add_argument('out_tractogram',
                   help='Output tractogram file. Note: data_per_point will be '
                        'discarded, if any!')

    p.add_argument('--label_ids', nargs=2, type=int,
                   help='List of labels indices to use to cut '
                        'streamlines (2 values).')
    p.add_argument('--resample', dest='step_size', type=float, default=None,
                   help='Resample streamlines to a specific step-size in mm '
                        '[%(default)s].')
    p.add_argument('--min_length', type=float, default=20,
                   help='Minimum length of streamlines to keep (in mm) '
                        '[%(default)s].')

    add_compression_arg(p)
    add_overwrite_arg(p)
    add_processes_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram, optional=[args.mask,
                                                              args.label,
                                                              args.reference])
    assert_outputs_exist(parser, args, args.out_tractogram)
    assert_headers_compatible(parser, args.in_tractogram,
                              optional=[args.mask,
                                        args.label],
                              reference=args.reference)

    # Loading
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    # Streamlines must be in voxel space to deal correctly with bounding box.
    sft.to_vox()
    sft.to_corner()
    # Resample streamlines to a specific step-size in mm. May impact the
    # cutting process.
    if args.step_size is not None:
        sft = resample_streamlines_step_size(sft, args.step_size)

    if len(sft.streamlines) == 0:
        parser.error('Input tractogram is empty.')

    # Mask scenario. Streamlines outside of the mask will be cut.
    if args.mask:
        mask_img = nib.load(args.mask)
        binary_mask = get_data_as_mask(mask_img)

        new_sft = cut_outside_of_mask_streamlines(sft, binary_mask,
                                                  min_len=args.min_length,
                                                  processes=args.nbr_processes)

    # Label scenario. The script will cut streamlines so they are going from
    # label 1 to label 2.
    else:
        label_img = nib.load(args.label)
        label_data = get_data_as_labels(label_img)

        new_sft = cut_between_mask_two_blobs_streamlines(
            sft, label_data, args.label_ids)

    # Saving
    if len(new_sft) == 0:
        logging.warning('No streamline intersected the provided mask. '
                        'Saving empty tractogram.')
    # Compress streamlines if requested
    elif args.compress_th:
        compressed_strs = [compress_streamlines(
            s, args.compress_th) for s in new_sft.streamlines]
        new_sft = StatefulTractogram.from_sft(
            compressed_strs, sft, data_per_streamline=sft.data_per_streamline)

    save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
