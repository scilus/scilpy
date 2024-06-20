#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze bundles at the fixel level using peaks and bundles (.trk). If the
bundles files are names as {bundle_name}.trk, simply use the --in_bundles
argument with --in_bundles_names. If it is not the case or you want other names
to be saved, please use --in_bundles_names to provide bundles names IN THE SAME
ORDER as the inputed bundles.

The script produces various output:

    - bundles_LUT.txt : array of (N) bundle names
      Lookup table (LUT) to know the order of the bundles in the various
      outputs. When an output contains results for all bundles, it is always
      the last dimension of the np.ndarray and it follows the order of the
      lookup table.

    - fixel_density_maps.nii.gz : np.ndarray (x, y, z, 5, N)
      For each voxel, it gives the density of bundles associated with each
      of the 5 fixels. If the normalization is chosen as the voxel-type, then
      the sum of the density over a voxel is 1. If the normalization is chosen
      as the fixel-type, then the sum of the density over each fixel is 1, so
      the sum over a voxel will be higher than 1 (except in the single-fiber
      case). The density maps can be computed using the streamline count, or
      any streamline weighting like COMMIT or SIF, through the
      data_per_streamline.
    
    - fixel_density_masks.nii.gz : np.ndarray (x, y, z, 5, N)
      For each voxel, it gives whether or not each bundle is associated
      with each of the 5 fixels. In other words, it is a masked version of
      fixel_density_maps, using two different thresholds. First, the absolute
      threshold (abs_thr) is applied on the maps before the normalization,
      either on the number of streamlines or the custom weight. Second, after
      the normalization, the relative threshold (rel_thr) is applied on the
      maps as a minimal value of density to be counted as an association.

    - voxel_density_maps.nii.gz : np.ndarray (x, y, z, N)
      For each voxel, it gives the density of each bundle within the voxel,
      regardless of fixels. In other words, it gives the fraction of each
      bundle per voxel. This is only outputed if the normalization of the maps
      is chosen as the voxel-type, because the fixel-type does not translate to
      meaningful results when summed into a voxel.

    - voxel_density_masks.nii.gz : np.ndarray (x, y, z, N)
      For each voxel, it gives whether or not each bundle is present. This is
      computed from fixel_density_masks, so the same thresholds prevail.

    - nb_bundles_per_fixel.nii.gz : np.ndarray (x, y, z)
      For each voxel, it gives the number of bundles associated with each of
      the 5 fixels.

    - nb_bundles_per_voxel.nii.gz : np.ndarray (x, y, z)
      For each voxel, it gives the number of bundles within the voxel. This
      accounts for bundles that might be associated with more than one fixel,
      so no bundle is counted more than once in a voxel.

    If the split_bundles argument is given, the script will also save the
    fixel_density_maps and fixel_density_masks separated by bundles, with
    names fixel_density_map_{bundle_name}.nii.gz and
    fixel_density_mask_{bundle_name}.nii.gz. 
    These will have the shape (x, y, z, 5).

    If the split_fixels argument is given, the script will also save the
    fixel_density_maps and fixel_density_masks separated by fixels, with
    names fixel_density_map_f{fixel_id}.nii.gz and
    fixel_density_mask_f{fixel_id}.nii.gz.
    These will have the shape (x, y, z, N).

    If the single_bundle argument is given, the script will also save the
    single-fiber single-bundle masks, which are obtained by selecting the
    voxels where only one bundle and one fiber (fixel) are present. There will
    be one single_bundle_mask_{bundle_name}.nii.gz per bundle, and a whole WM
    version single_bundle_mask_WM.nii.gz.
    These will have the shape (x, y, z).

"""

import argparse
import nibabel as nib
import numpy as np
import logging
from pathlib import Path

from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             assert_headers_compatible, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg)
from scilpy.tractanalysis.fixel_density import (fixel_density, maps_to_masks)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_peaks',
                   help='Path of the peaks. The peaks are expected to be '
                        'given as unit directions. \nTo get these from fODF '
                        'or SH data, use the script scil_fodf_metrics.py '
                        '\nwith the abs_peaks_and_values option.')

    p.add_argument('--in_bundles', nargs='+', action='append', required=True,
                   help='List of paths of the bundles (.trk) to analyze.')

    p.add_argument('--in_bundles_names', nargs='+', action='append',
                   help='List of the names of the bundles, in the same order '
                        'as they were given. \nIf this argument is not used, '
                        'the script assumes that the name of the bundle \nis '
                        'its filename without extensions.')

    g = p.add_argument_group(title='Mask parameters')

    g.add_argument('--abs_thr', default=1, type=int,
                   help='Value of density maps threshold to obtain density '
                        'masks, in number of streamlines. \nAny number of '
                        'streamlines above or equal this value will pass '
                        'the absolute threshold test [%(default)s].')

    g.add_argument('--rel_thr', default=0.01, type=float,
                   help='Value of density maps threshold to obtain density '
                        'masks, as a ratio of the normalized density '
                        '\nAny normalized density above or equal to '
                        'this value will pass the relative threshold test. '
                        '\nMust be between 0 and 1 [%(default)s].')

    g.add_argument('--norm', default="voxel", choices=["fixel", "voxel"],
                   help='Way of normalizing the density maps. If fixel, '
                        'will normalize the maps per fixel, \nin each voxel. '
                        'If voxel, will normalize the maps per voxel. '
                        '[%(default)s]')

    p.add_argument('--max_theta', default=45,
                   help='Maximum angle between streamline and peak to be '
                        'associated [%(default)s].')

    p.add_argument('--split_bundles', action='store_true',
                   help='If set, save the density maps for each bundle '
                        'separately \ninstead of all in one file.')

    p.add_argument('--split_fixels', action='store_true',
                   help='If set, save the density maps for each fixel '
                        'separately \ninstead of all in one file.')

    p.add_argument('--single_bundle', action='store_true',
                   help='If set, will save the single-fiber single-bundle '
                        'masks as well. \nThese are obtained by '
                        'selecting the voxels where only one bundle is '
                        'present \n(and one fiber/fixel).')

    add_overwrite_arg(p)
    add_processes_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_peaks] + args.in_bundles[0])
    assert_outputs_exist(parser, args, ["bundles_LUT.txt",
                                        "fixel_density_maps.nii.gz",
                                        "fixel_density_masks.nii.gz",
                                        "voxel_density_masks.nii.gz",
                                        "nb_bundles_per_fixel.nii.gz",
                                        "nb_bundles_per_voxel.nii.gz"])
    assert_headers_compatible(parser, [args.in_peaks] + args.in_bundles[0])

    if args.rel_thr < 0 or args.rel_thr > 1:
        parser.error("Argument rel_thr must be a value between 0 and 1.")

    # Load the data
    peaks_img = nib.load(args.in_peaks)
    peaks = peaks_img.get_fdata()
    affine = peaks_img.affine

    # Compute NuFo single-fiber from peaks
    if args.single_bundle:
        is_first_peak = np.sum(peaks[..., 0:3], axis=-1) != 0
        is_second_peak = np.sum(peaks[..., 3:6], axis=-1) == 0
        nufo_sf = np.logical_and(is_first_peak, is_second_peak)

    # Extract bundles and names
    bundles = []
    bundles_names = []
    for bundle in args.in_bundles[0]:
        bundles.append(bundle)
        bundles_names.append(Path(bundle).name.split(".")[0])
    if args.in_bundles_names:  # If names are given
        bundles_names = args.in_bundles_names[0]

    # Compute fixel density maps and masks
    fixel_density_maps = fixel_density(peaks, bundles, args.max_theta,
                                       nbr_processes=args.nbr_processes)

    fixel_density_masks, fixel_density_maps = maps_to_masks(fixel_density_maps,
                                                            args.abs_thr,
                                                            args.rel_thr,
                                                            args.norm,
                                                            len(bundles))

    # Compute number of bundles per fixel
    nb_bundles_per_fixel = np.sum(fixel_density_masks, axis=-1)
    # Compute voxel density maps
    voxel_density_maps = np.sum(fixel_density_maps, axis=-2)
    # Compute voxel density masks
    # Since a bundle can be present twice in a single voxel by being associated
    # with more than one fixel, we count the presence of a bundle if > 0.
    voxel_density_masks = np.where(np.sum(fixel_density_masks, axis=-2) > 0,
                                   1, 0)
    # Compute number of bundles per voxel by taking the sum of the mask
    nb_bundles_per_voxel = np.sum(voxel_density_masks, axis=-1)

    # Save all results
    for i, bundle_name in enumerate(bundles_names):
        if args.split_bundles:  # Save the maps and masks for each bundle
            nib.save(nib.Nifti1Image(fixel_density_maps[..., i], affine),
                     "fixel_density_map_{}.nii.gz".format(bundle_name))
            nib.save(nib.Nifti1Image(fixel_density_masks[..., i], affine),
                     "fixel_density_mask_{}.nii.gz".format(bundle_name))

        if args.single_bundle:
            # Single-fiber single-bundle voxels
            one_bundle_per_voxel = nb_bundles_per_voxel == 1
            # Making sure we also have single-fiber voxels only
            one_bundle_per_voxel *= nufo_sf
            # Save a single-fiber single-bundle mask for the whole WM
            nib.save(nib.Nifti1Image(one_bundle_per_voxel.astype(np.uint8),
                                     affine),
                     "single_bundle_mask_WM.nii.gz")
            # Save a single-fiber single-bundle mask for each bundle
            bundle_mask = fixel_density_masks[..., 0, i] * one_bundle_per_voxel
            nib.save(nib.Nifti1Image(bundle_mask.astype(np.uint8), affine),
                     "single_bundle_mask_{}.nii.gz".format(bundle_name))

    if args.split_fixels:  # Save the maps and masks for each fixel
        for i in range(5):
            nib.save(nib.Nifti1Image(fixel_density_maps[..., i, :], affine),
                     "fixel_density_map_f{}.nii.gz".format(i + 1))
            nib.save(nib.Nifti1Image(fixel_density_masks[..., i, :], affine),
                     "fixel_density_mask_f{}.nii.gz".format(i + 1))

    # Save bundles lookup table to know the order of the bundles
    bundles_idx = np.arange(0, len(bundles_names), 1)
    lookup_table = np.array([bundles_names, bundles_idx])
    np.savetxt("bundles_LUT.txt", lookup_table, fmt='%s')

    # Save full fixel density maps, all fixels and bundles combined
    nib.save(nib.Nifti1Image(fixel_density_maps, affine),
             "fixel_density_maps.nii.gz")

    # Save full fixel density masks, all fixels and bundles combined
    nib.save(nib.Nifti1Image(fixel_density_masks, affine),
             "fixel_density_masks.nii.gz")
    
    # Save full voxel density maps and masks
    if args.norm == "voxel":  # If fixel, the voxel maps do not mean anything
        nib.save(nib.Nifti1Image(voxel_density_maps, affine),
                 "voxel_density_maps.nii.gz")
    nib.save(nib.Nifti1Image(voxel_density_masks.astype(np.uint8), affine),
             "voxel_density_masks.nii.gz")

    # Save number of bundles per fixel and per voxel
    nib.save(nib.Nifti1Image(nb_bundles_per_fixel.astype(np.uint8), affine),
             "nb_bundles_per_fixel.nii.gz")
    nib.save(nib.Nifti1Image(nb_bundles_per_voxel.astype(np.uint8), affine),
             "nb_bundles_per_voxel.nii.gz")


if __name__ == "__main__":
    main()
