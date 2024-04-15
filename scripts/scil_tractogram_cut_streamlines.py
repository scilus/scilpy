#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filters streamlines and only keeps the parts of streamlines within or
between the ROIs. Two options are available.

Input mask:

The mask has either 1 entity/blob or
2 entities/blobs (does not support disconnected voxels).
The option --biggest_blob can help if you have such a scenario.

The 1 entity scenario will 'trim' the streamlines so their longest segment is
within the bounding box or a binary mask.

The 2 entities scenario will cut streamlines so their segment are within the
bounding box or going from binary mask #1 to binary mask #2.

Input label:

The label MUST contain 2 labels different from zero.
Label values could be anything.
The script will cut streamlines going from label 1 to label 2.

Both inputs and scenarios will erase data_per_point and data_per_streamline.

Formerly: scil_cut_streamlines.py
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import compress_streamlines
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from scilpy.image.labels import get_data_as_labels
from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
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
                    help='Binary mask containing either 1 or 2 blobs.')
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
    p.add_argument('--biggest_blob', action='store_true',
                   help='Use the biggest entity and force the 1 ROI scenario.')

    add_compression_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

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

    if args.step_size is not None:
        sft = resample_streamlines_step_size(sft, args.step_size)

    if len(sft.streamlines) == 0:
        parser.error('Input tractogram is empty.')

    if args.mask:
        mask_img = nib.load(args.mask)
        binary_mask = get_data_as_mask(mask_img)

        bundle_disjoint, _ = ndi.label(binary_mask)
        unique, count = np.unique(bundle_disjoint, return_counts=True)
        if args.biggest_blob:
            val = unique[np.argmax(count[1:])+1]
            binary_mask[bundle_disjoint != val] = 0
            unique = [0, val]
        if len(unique) == 2:
            logging.info('The provided mask has 1 entity '
                         'cut_outside_of_mask_streamlines function selected.')
            new_sft = cut_outside_of_mask_streamlines(sft, binary_mask)
        elif len(unique) == 3:
            logging.info('The provided mask has 2 entity '
                         'cut_between_mask_two_blobs_streamlines '
                         'function selected.')
            new_sft = cut_between_mask_two_blobs_streamlines(sft, binary_mask)
        else:
            logging.warning('The provided mask has MORE THAN 2 entity '
                            'cut_between_mask_two_blobs_streamlines function '
                            'selected. This may cause problems with '
                            'the outputed streamlines.'
                            ' Please inspect the output carefully.')
            new_sft = cut_between_mask_two_blobs_streamlines(sft, binary_mask)
    else:
        label_img = nib.load(args.label)
        label_data = get_data_as_labels(label_img)

        if args.label_ids:
            unique_vals = args.label_ids
        else:
            unique_vals = np.unique(label_data[label_data != 0])
            if len(unique_vals) != 2:
                parser.error('More than two values in the label file, '
                             'please use --label_ids to select '
                             'specific label ids.')

        label_data_1 = np.copy(label_data)
        mask = label_data_1 != unique_vals[0]
        label_data_1[mask] = 0

        label_data_2 = np.copy(label_data)
        mask = label_data_2 != unique_vals[1]
        label_data_2[mask] = 0

        new_sft = cut_between_mask_two_blobs_streamlines(sft, label_data_1,
                                                         binary_mask_2=label_data_2)

    # Saving
    if len(new_sft) == 0:
        logging.warning('No streamline intersected the provided mask. '
                        'Saving empty tractogram.')
    elif args.compress_th:
        compressed_strs = [compress_streamlines(
            s, args.compress_th) for s in new_sft.streamlines]
        new_sft = StatefulTractogram.from_sft(
            compressed_strs, sft, data_per_streamline=sft.data_per_streamline)

    save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
