#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uniformize streamlines' endpoints according to a defined axis.
Useful for tractometry or models creation.

The --auto option will automatically calculate the main orientation.
If the input bundle is poorly defined, it is possible heuristic will be wrong.

The default is to flip each streamline so their first point's coordinate in the
defined axis is smaller than their last point (--swap does the opposite).

The --target_roi option will use the barycenter of the target mask to define
the axis. The target mask can be a binary mask or an atlas. If an atlas is
used, labels are expected in the form of --target_roi atlas.nii.gz 2 3 5:7.

Formerly: scil_uniformize_streamlines_endpoints.py
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram
import nibabel as nib

from scilpy.image.labels import get_data_as_labels
from scilpy.io.image import merge_labels_into_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_outputs_exist,
                             assert_inputs_exist, assert_headers_compatible)
from scilpy.tractanalysis.bundle_operations import \
    uniformize_bundle_sft, uniformize_bundle_sft_using_mask


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Input path of the tractography file.')
    p.add_argument('out_bundle',
                   help='Output path of the uniformized file.')

    method = p.add_mutually_exclusive_group(required=True)
    method.add_argument('--axis', choices=['x', 'y', 'z'],
                        help='Match endpoints of the streamlines along this '
                        'axis.\nSUGGESTION: Commissural = x, Association = y, '
                        'Projection = z')
    method.add_argument('--auto', action='store_true',
                        help='Match endpoints of the streamlines along an '
                             'automatically determined axis.')
    method.add_argument('--centroid', metavar='tractogram',
                        help='Match endpoints of the streamlines to align it '
                             'to a reference unique streamline (centroid).')
    method.add_argument('--target_roi', nargs='+',
                        help='Provide a target ROI: either a binary mask or a '
                             'label map and the labels to use.\n'
                             'Will align heads to be closest to the mask '
                             'barycenter.\n'
                             '(atlas: if no labels are provided, all labels '
                             'will be used.')
    p.add_argument('--swap', action='store_true',
                   help='Swap head <-> tail convention. '
                        'Can be useful when the reference is not in RAS.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    roi_file = args.target_roi[0] if args.target_roi else None
    assert_inputs_exist(parser, args.in_bundle, [roi_file, args.reference])
    assert_outputs_exist(parser, args, args.out_bundle)
    assert_headers_compatible(parser, args.in_bundle, roi_file,
                              reference=args.reference)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    if args.auto:
        uniformize_bundle_sft(sft, None, swap=args.swap)

    if args.centroid:
        centroid_sft = load_tractogram_with_reference(parser, args,
                                                      args.centroid)
        uniformize_bundle_sft(sft, args.axis, swap=args.swap,
                              ref_bundle=centroid_sft)

    if args.target_roi:
        img = nib.load(args.target_roi[0])
        atlas = get_data_as_labels(img)
        if len(args.target_roi) == 1:
            mask = atlas > 0
        else:
            mask = merge_labels_into_mask(atlas, " ".join(args.target_roi[1:]))

        # Uncomment if the user wants to filter the streamlines
        # sft, _ = filter_grid_roi(sft, mask, 'either_end', False)
        uniformize_bundle_sft_using_mask(sft, mask, swap=args.swap)

    if args.axis:
        uniformize_bundle_sft(sft, args.axis, swap=args.swap)

    save_tractogram(sft, args.out_bundle)


if __name__ == "__main__":
    main()
