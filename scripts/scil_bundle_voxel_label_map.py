#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_trk_in_voxel_space
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_verbose_arg,
                             add_reference)
from scilpy.tractanalysis import compute_robust_tract_counts_map
from scilpy.tractometry.distance_to_centroid import min_dist_to_centroid


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute (upsampled) Nifti label image from bundle and '
                    'centroid. Each voxel will have the label of its '
                    'nearest centroid point',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('bundle',
                   help='Fiber bundle file.')
    p.add_argument('centroid_streamline',
                   help='Centroid streamline corresponding to bundle.')

    add_reference(p)

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
                        [args.bundle, args.centroid_streamline,
                         args.reference])
    assert_outputs_exist(parser, args, args.output_map)

    bundle_tractogram_file = nib.streamlines.load(args.bundle)
    centroid_tractogram_file = nib.streamlines.load(args.centroid_streamline)
    if int(bundle_tractogram_file.header['nb_streamlines']) == 0:
        logger.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return

    if int(centroid_tractogram_file.header['nb_streamlines']) != 1:
        logger.warning('Centroid file {} should contain one streamline. '
                       'Skipping'.format(args.centroid_streamline))
        return

    ref_img = nib.load(args.reference)
    bundle_streamlines_vox = load_trk_in_voxel_space(
        bundle_tractogram_file, anat=ref_img)
    bundle_streamlines_vox._data *= args.upsample

    centroid_streamlines_vox = load_trk_in_voxel_space(
        centroid_tractogram_file, anat=ref_img)
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
