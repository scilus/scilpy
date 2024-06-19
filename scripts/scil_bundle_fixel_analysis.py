#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze bundles at the fixel level, producing various results:

-
-
-

"""

import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_reference_arg)
from scilpy.tractanalysis.fixel_density import compute_fixel_density


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_peaks',
                   help='Path of the fODF peaks. The peaks are expected to be '
                        'given as unit directions.')
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')

    p.add_argument('--in_bundles', nargs='+', default=[],
                   action='append', required=True,
                   help='Path to the bundle trk for where to analyze.')

    p.add_argument('--abs_thr', default=1, type=int,
                   help='Value of density maps threshold to obtain density '
                        'masks, in number of streamlines. Any number of '
                        'streamlines above or equal to this value will pass '
                        'the absolute threshold test [%(default)s].')

    p.add_argument('--rel_thr', default=0.01, type=float,
                   help='Value of density maps threshold to obtain density '
                        'masks, as a ratio of the normalized density. Any '
                        'normalized density above or equal to this value will '
                        'pass the relative threshold test [%(default)s].')

    p.add_argument('--max_theta', default=45,
                   help='Maximum angle between streamline and peak to be '
                        'associated [%(default)s].')

    p.add_argument('--separate_bundles', action='store_true',
                   help='If set, save the density maps for each bundle '
                        'separately instead of all in one file.')

    p.add_argument('--save_masks', action='store_true',
                   help='If set, save the density masks for each bundle.')

    p.add_argument('--select_single_bundle', action='store_true',
                   help='If set, select the voxels where only one bundle is '
                        'present and save the corresponding masks '
                        '(if save_masks), separated by bundles.')

    p.add_argument('--norm', default="voxel", choices=["fixel", "voxel"],
                   help='Way of normalizing the density maps. If fixel, '
                        'will normalize the maps per fixel, in each voxel. '
                        'If voxel, will normalize the maps per voxel. '
                        '[%(default)s].')

    add_overwrite_arg(p)
    add_processes_arg(p)
    add_reference_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    out_folder = Path(args.out_folder)

    # Load the data
    peaks_img = nib.load(args.in_peaks)
    peaks = peaks_img.get_fdata()
    affine = peaks_img.affine

    # Compute NuFo SF from peaks
    if args.select_single_bundle:
        is_first_peak = np.sum(peaks[..., 0:3], axis=-1) != 0
        is_second_peak = np.sum(peaks[..., 3:6], axis=-1) == 0
        nufo_sf = np.logical_and(is_first_peak, is_second_peak)

    bundles = []
    bundles_names = []
    for bundle in args.in_bundles[0]:
        bundles.append(bundle)
        bundles_names.append(Path(bundle).name.split(".")[0])

    fixel_density_maps = compute_fixel_density(peaks, bundles, args.max_theta,
                                               nbr_processes=args.nbr_processes)

    # This applies a threshold on the number of streamlines.
    fixel_density_masks_abs = fixel_density_maps >= args.abs_thr

    # Normalizing the density maps
    fixel_sum = np.sum(fixel_density_maps, axis=-1)
    voxel_sum = np.sum(fixel_sum, axis=-1)

    for i, (bundle, bundle_name) in enumerate(zip(bundles, bundles_names)):

        if args.norm == "voxel":
            fixel_density_maps[..., 0, i] /= voxel_sum
            fixel_density_maps[..., 1, i] /= voxel_sum
            fixel_density_maps[..., 2, i] /= voxel_sum
            fixel_density_maps[..., 3, i] /= voxel_sum
            fixel_density_maps[..., 4, i] /= voxel_sum

        elif args.norm == "fixel":
            fixel_density_maps[..., i] /= fixel_sum

        if args.separate_bundles:
            nib.save(nib.Nifti1Image(fixel_density_maps[..., i], affine),
                     out_folder / "fixel_density_map_{}.nii.gz".format(bundle_name))

    if not args.separate_bundles:
        nib.save(nib.Nifti1Image(fixel_density_maps, affine),
                 out_folder / "fixel_density_maps.nii.gz")
        bundles_idx = np.arange(0, len(bundles_names), 1)
        lookup_table = np.array([bundles_names, bundles_idx])
        np.savetxt(out_folder / "bundles_lookup_table.txt",
                   lookup_table, fmt='%s')

    # This applies a threshold on the normalized density (percentage)
    fixel_density_masks_rel = fixel_density_maps >= args.rel_thr

    # Compute the fixel density masks from the rel and abs versions
    fixel_density_masks = fixel_density_masks_rel * fixel_density_masks_abs

    # Compute number of bundles per fixel
    nb_bundles_per_fixel = np.sum(fixel_density_masks, axis=-1)
    # Compute a mask of the present of each bundle
    nb_unique_bundles_per_fixel = np.where(np.sum(fixel_density_masks,
                                                  axis=-2) > 0, 1, 0)
    # Compute number of bundles per fixel by taking the sum of the mask
    nb_bundles_per_voxel = np.sum(nb_unique_bundles_per_fixel, axis=-1)

    if args.save_masks:
        for i, (bundle, bundle_name) in enumerate(zip(bundles, bundles_names)):

            if args.separate_bundles:
                nib.save(nib.Nifti1Image(fixel_density_masks[..., i].astype(np.uint8), affine),
                         out_folder / "fixel_density_mask_{}.nii.gz".format(bundle_name))

            if args.select_single_bundle:
                # Single-fiber single-bundle voxels
                single_bundle_per_voxel = nb_bundles_per_voxel == 1
                # Making sure we also have single-fiber voxels only
                single_bundle_per_voxel *= nufo_sf

                nib.save(nib.Nifti1Image(single_bundle_per_voxel.astype(np.uint8),
                                         affine),
                         out_folder / "bundle_mask_only_WM.nii.gz")

                bundle_mask = fixel_density_masks[..., 0, i] * single_bundle_per_voxel
                nib.save(nib.Nifti1Image(bundle_mask.astype(np.uint8), affine),
                         out_folder / "bundle_mask_only_{}.nii.gz".format(bundle_name))

        if not args.separate_bundles:
            nib.save(nib.Nifti1Image(fixel_density_masks.astype(np.uint8),
                                     affine),
                     out_folder / "fixel_density_masks.nii.gz")

    nib.save(nib.Nifti1Image(nb_bundles_per_fixel.astype(np.uint8),
             affine), out_folder / "nb_bundles_per_fixel.nii.gz")

    nib.save(nib.Nifti1Image(nb_bundles_per_voxel.astype(np.uint8),
             affine), out_folder / "nb_bundles_per_voxel.nii.gz")


if __name__ == "__main__":
    main()
