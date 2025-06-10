#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cut streamlines using a binary mask or two labels.

--mask: Binary mask. Streamlines outside of the mask will be cut. Three options
are available:

    Default: Will cut the streamlines according to the mask. New streamlines
    may be generated if the mask is disjoint.

    --keep_longest: Will keep the longest segment of the streamline that is
    within the mask. No new streamlines will be generated.

    --trim_endpoints: Will only remove the endpoints of the streamlines that
    are outside the mask. The middle part of the streamline may go
    outside the mask, to compensate for hole in the mask for example. No new
    streamlines will be generated.

--label: Label containing 2 blobs. Streamlines will be cut so they go from the
first label region to the second label region. The two blobs must be disjoint.

    Default: The streamline will start with the first point of the last segment
    of the streamline in the first label and end with the last point of the
    first segment of the streamline in the second label.

    --one_point_in_roi:
        The streamline will start with the last point of the last segment of
        the streamline in the first label and end with the first point in the
        second label. The streamline will be cut at the boundary of the labels.

    --no_point_in_roi:
        The streamline will be cut at the boundary of the labels. No point will
        be kept in the labels.

Both scenarios will erase data_per_point and data_per_streamline. Streamlines
will be extended so they reach the boundary of the mask or the two labels,
therefore won't be equal to the input streamlines.

To generate a label map from a binary mask, you can use the following command:
    scil_labels_from_mask.py

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
    cut_streamlines_with_mask, cut_streamlines_between_labels, \
    CuttingStyle
from scilpy.tractograms.streamline_operations import \
    resample_streamlines_step_size
from scilpy.version import version_string

# Mapping the arguments to the cutting style
# (keep_longest, trim_endpoints) -> CuttingStyle
args_to_style = {(False, False): CuttingStyle.DEFAULT,
                 (True, False): CuttingStyle.KEEP_LONGEST,
                 (False, True): CuttingStyle.TRIM_ENDPOINTS}


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractogram',
                   help='Input tractogram file.')

    g1 = p.add_mutually_exclusive_group(required=True)
    g1.add_argument('--mask',
                    help='Binary mask.')
    g1.add_argument('--labels',
                    help='Label containing 2 blobs.')

    p.add_argument('out_tractogram',
                   help='Output tractogram file. Note: data_per_point and '
                        'data_per_streamline will be discarded.')
    p.add_argument('--label_ids', nargs=2, type=int,
                   help='List of labels indices to use to cut '
                        'streamlines (2 values).')
    p.add_argument('--resample', dest='step_size', type=float, default=None,
                   help='Resample streamlines to a specific step-size in mm '
                        '[%(default)s].')
    p.add_argument('--min_length', type=float, default=0,
                   help='Minimum length of streamlines to keep (in mm) '
                        '[%(default)s].')

    g = p.add_argument_group('Cutting options', 'Options for cutting '
                             'streamlines with --mask.')
    g2 = g.add_mutually_exclusive_group()
    g2.add_argument('--keep_longest', action='store_true',
                    help='If set, will keep the longest segment of the '
                         'streamline that is within the mask.')
    g2.add_argument('--trim_endpoints', action='store_true',
                    help='If set, will only remove the endpoints of the '
                         'streamlines that are outside the mask.')

    g1 = p.add_argument_group('Cutting options', 'Options for cutting '
                              'streamlines with --labels.')
    g3 = g1.add_mutually_exclusive_group()
    g3.add_argument('--one_point_in_roi', action='store_true',
                    help='If set, will keep one point in each label.')
    g3.add_argument('--no_point_in_roi', action='store_true',
                    help='If set, will not keep any point in the labels.')

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
                                                              args.labels,
                                                              args.reference])
    assert_outputs_exist(parser, args, args.out_tractogram)
    assert_headers_compatible(parser, args.in_tractogram,
                              optional=[args.mask,
                                        args.labels],
                              reference=args.reference)

    if args.labels and (args.keep_longest or args.trim_endpoints):
        parser.error('Cannot use --keep_longest or --trim_endpoints with '
                     'labels.')
    elif args.mask and (args.one_point_in_roi or args.no_point_in_roi):
        parser.error('Cannot use --one_point_in_roi or --no_point_in_roi with '
                     'mask.')

    # Loading
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    # Resample streamlines to a specific step-size in mm. May impact the
    # cutting process.
    if args.step_size is not None:
        sft = resample_streamlines_step_size(sft, args.step_size)

    if len(sft.streamlines) == 0:
        parser.error('Input tractogram is empty.')

    # Mask scenario, either keeping the longest segment of the streamline that
    # is in the mask, trimming the endpoints of the streamlines outside of the
    # mask or cutting the streamlines outside of the mask.
    if args.mask:
        style = args_to_style[args.keep_longest, args.trim_endpoints]
        mask_img = nib.load(args.mask)
        binary_mask = get_data_as_mask(mask_img)

        new_sft = cut_streamlines_with_mask(
            sft, binary_mask, cutting_style=style,
            min_len=args.min_length, processes=args.nbr_processes)
    # Label scenario. The script will cut streamlines so they are going from
    # label 1 to label 2.
    else:
        label_img = nib.load(args.labels)
        label_data = get_data_as_labels(label_img)

        new_sft = cut_streamlines_between_labels(
            sft, label_data, args.label_ids, min_len=args.min_length,
            one_point_in_roi=args.one_point_in_roi,
            no_point_in_roi=args.no_point_in_roi,
            processes=args.nbr_processes)

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
