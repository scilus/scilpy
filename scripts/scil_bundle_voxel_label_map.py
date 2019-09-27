#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging

import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_verbose_arg,
                             add_reference)
from scilpy.tractanalysis import compute_tract_counts_map
from scilpy.tractometry.distance_to_centroid import min_dist_to_centroid


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute (upsampled) Nifti label image from bundle and '
                    'centroid. Each voxel will have the label of its '
                    'nearest centroid point',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')
    p.add_argument('in_centroid',
                   help='Centroid streamline corresponding to bundle.')
    p.add_argument('reference',
                   help='Nifti reference image.')
    p.add_argument('output_map',
                   help='Nifti image with corresponding labels.')
    p.add_argument('--upsample',
                   type=float, default=2,
                   help='Upsample reference grid by this factor. '
                        '[%(default)s]')

    add_overwrite_arg(p)

    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser,
                        [args.in_bundle, args.in_centroid],
                        optionnal=args.reference)
    assert_outputs_exist(parser, args, args.output_map)

    sft_bundle = load_tractogram_with_reference(parser, args,
                                                args.in_bundle)
    sft_centroid = load_tractogram_with_reference(parser, args,
                                                args.in_centroid)

    if len(sft_bundle.streamlines) == 0:
        logging.warning('Empty bundle file {}. '
                        'Skipping'.format(args.in_bundle))
        return

    if len(sft_centroid.streamlines) == 0:
        logging.warning('Centroid file {} should contain one streamline. '
                        'Skipping'.format(args.in_centroid))
        return

    ref_img = nib.load(args.reference)

    sft_bundle.to_vox()
    bundle_streamlines_vox = sft.streamlines
    bundle_streamlines_vox._data *= args.upsample

    sft_centroid.to_vox()
    centroid_streamlines_vox = sft_centroid.streamlines
    centroid_streamlines_vox._data *= args.upsample

    upsampled_shape = [s * args.upsample for s in ref_img.shape]
    tdi_mask = compute_robust_tract_counts_map(bundle_streamlines_vox,
                                               upsampled_shape) > 0

    tdi_mask_nzr = np.nonzero(tdi_mask)
    tdi_mask_nzr_ind = np.transpose(tdi_mask_nzr)

    min_dist_ind, _ = min_dist_to_centroid(tdi_mask_nzr_ind,
                                           centroid_streamlines_vox[0])

    # Save the (upscaled) labels mask
    labels_mask = np.zeros(tdi_mask.shape)
    labels_mask[tdi_mask_nzr] = min_dist_ind + 1  # 0 is background value
    rescaled_affine = ref_img.affine
    rescaled_affine[:3, :3] /= args.upsample
    labels_img = nib.Nifti1Image(labels_mask, rescaled_affine)
    upsampled_spacing = ref_img.header['pixdim'][1:4] / args.upsample
    labels_img.header.set_zooms(upsampled_spacing)
    nib.save(labels_img, args.output_map)


if __name__ == '__main__':
    main()
