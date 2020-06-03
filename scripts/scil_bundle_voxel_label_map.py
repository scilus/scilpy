#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute label image (Nifti) from bundle and centroid.
Each voxel will have the label of its nearest centroid point.

The number of labels will be the same as the centroid's number of points.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_verbose_arg)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.distance_to_centroid import min_dist_to_centroid


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')
    p.add_argument('in_centroid',
                   help='Centroid streamline corresponding to bundle.')
    p.add_argument('out_map',
                   help='Nifti image with corresponding labels.')
    p.add_argument('--upsample', type=float, default=1,
                   help='Upsample reference grid by this factor. '
                        '[%(default)s]')

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser,
                        [args.in_bundle, args.in_centroid],
                        optional=args.reference)
    assert_outputs_exist(parser, args, args.out_map)

    sft_bundle = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft_centroid = load_tractogram_with_reference(parser, args,
                                                  args.in_centroid)

    if not len(sft_bundle.streamlines):
        logging.error('Empty bundle file {}. '
                      'Skipping'.format(args.in_bundle))
        raise ValueError

    if not len(sft_centroid.streamlines):
        logging.error('Centroid file {} should contain one streamline. '
                      'Skipping'.format(args.in_centroid))
        raise ValueError

    sft_bundle.to_vox()
    bundle_streamlines_vox = sft_bundle.streamlines
    bundle_streamlines_vox._data *= args.upsample

    sft_centroid.to_vox()
    centroid_streamlines_vox = sft_centroid.streamlines
    centroid_streamlines_vox._data *= args.upsample

    upsampled_shape = [s * args.upsample for s in sft_bundle.dimensions]
    tdi_mask = compute_tract_counts_map(bundle_streamlines_vox,
                                        upsampled_shape) > 0

    tdi_mask_nzr = np.nonzero(tdi_mask)
    tdi_mask_nzr_ind = np.transpose(tdi_mask_nzr)

    min_dist_ind, _ = min_dist_to_centroid(tdi_mask_nzr_ind,
                                           centroid_streamlines_vox[0])

    # Save the (upscaled) labels mask
    labels_mask = np.zeros(tdi_mask.shape)
    labels_mask[tdi_mask_nzr] = min_dist_ind + 1  # 0 is background value
    rescaled_affine = sft_bundle.affine
    rescaled_affine[:3, :3] /= args.upsample
    labels_img = nib.Nifti1Image(labels_mask, rescaled_affine)
    upsampled_spacing = sft_bundle.voxel_sizes / args.upsample
    labels_img.header.set_zooms(upsampled_spacing)
    nib.save(labels_img, args.out_map)


if __name__ == '__main__':
    main()
