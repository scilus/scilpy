#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filters streamlines and only keeps the parts of streamlines within or
between the ROIs. The script accepts a single input mask, the mask has either
1 entity/blob or 2 entities/blobs (does not support disconnected voxels).
The option --biggest_blob can help if you have such a scenario.

The 1 entity scenario will 'trim' the streamlines so their longest segment is
within the bounding box or a binary mask.

The 2 entities scenario will cut streamlines so their segment are within the
bounding box or going from binary mask #1 to binary mask #2.

Both scenarios will erase data_per_point and data_per_streamline.
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import compress_streamlines
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
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
    p.add_argument('in_mask',
                   help='Binary mask containing either 1 or 2 blobs.')
    p.add_argument('out_tractogram',
                   help='Output tractogram file.')

    p.add_argument('--resample', dest='step_size', type=float, default=None,
                   help='Resample streamlines to a specific step-size in mm '
                        '[%(default)s].')
    p.add_argument('--compress', dest='error_rate', type=float, default=None,
                   help='Maximum compression distance in mm [%(default)s].')
    p.add_argument('--biggest_blob', action='store_true',
                   help='Use the biggest entity and force the 1 ROI scenario.')
    add_overwrite_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_inputs_exist(parser, [args.in_tractogram, args.in_mask])
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    if args.step_size is not None:
        sft = resample_streamlines_step_size(sft, args.step_size)

    mask_img = nib.load(args.in_mask)
    binary_mask = get_data_as_mask(mask_img)

    if not is_header_compatible(sft, mask_img):
        parser.error('Incompatible header between the tractogram and mask.')

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
                        'selected. This may cause problems with the outputed '
                        'streamlines. Please inspect the output carefully.')
        new_sft = cut_between_mask_two_blobs_streamlines(sft, binary_mask)

    if len(new_sft) == 0:
        logging.warning('No streamline intersected the provided mask. '
                        'Saving empty tractogram.')
    elif args.error_rate is not None:
        compressed_strs = [compress_streamlines(
            s, args.error_rate) for s in new_sft.streamlines]
        new_sft = StatefulTractogram.from_sft(compressed_strs, sft)

    save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
