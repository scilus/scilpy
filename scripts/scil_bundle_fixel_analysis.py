#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze bundles at the fixel level using peaks and bundles (.trk). If the
bundles files are names as {bundle_name}.trk, simply use the --in_bundles
argument without --in_bundles_names. If it is not the case or you want other
names to be saved, please use --in_bundles_names to provide bundles names
IN THE SAME ORDER as the inputed bundles.

The duration of the script depends heavily on the number of bundles, the number
of streamlines in bundles and the number of processors used. For a standard
tractoflow/rbx_flow output with ~30 bundles, it should not take over 30
minutes.

The script produces various output:

    - bundles_LUT.txt : array of (N) bundle names
      Lookup table (LUT) to know the order of the bundles in the various
      outputs. When an output contains results for all bundles, it is always
      the last dimension of the np.ndarray and it follows the order of the
      lookup table.

    - fixel_density_maps : np.ndarray (x, y, z, 5, N)
      For each voxel, it gives the density of bundles associated with each
      of the 5 fixels. If the normalization is chosen as the voxel-type, then
      the sum of the density over a voxel is 1. If the normalization is chosen
      as the fixel-type, then the sum of the density over each fixel is 1, so
      the sum over a voxel will be higher than 1 (except in the single-fiber
      case). If the normalization if chosen as none, no normalization is done.
      This can be useful to keep the information un-normalized.
      The density maps can be computed using the streamline count, or
      any streamline weighting like COMMIT or SIFT, through the
      data_per_streamline using the --dps_key argument. NOTE: This is a 5D file
      that will not be easily inputed to a regular viewer. Use --split_bundles
      or --split_fixels options to save the 4D versions for visualization.

    - fixel_density_masks : np.ndarray (x, y, z, 5, N)
      For each voxel, it gives whether or not each bundle is associated
      with each of the 5 fixels. In other words, it is a masked version of
      fixel_density_maps, using two different thresholds. First, the absolute
      threshold (abs_thr) is applied on the maps before the normalization,
      either on the number of streamlines or the streamline weights. Second,
      after the normalization, the relative threshold (rel_thr) is applied on
      the maps as a minimal value of density to be counted as an association.
      This does nothing if the normalization is set to none.
      NOTE: This is a 5D file that will not be easily inputed to a regular
      viewer. Use --split_bundles or --split_fixels options to save the 4D
      versions for visualization.

    - voxel_density_maps : np.ndarray (x, y, z, N)
      For each voxel, it gives the density of each bundle within the voxel,
      regardless of fixels. In other words, it gives the fraction of each
      bundle per voxel. This is only outputed if the normalization of the maps
      is chosen as the voxel-type or none-type, because the fixel-type does not
      translate to meaningful results when summed into a voxel.

    - voxel_density_masks: np.ndarray (x, y, z, N)
      For each voxel, it gives whether or not each bundle is present. This is
      computed from fixel_density_masks, so the same thresholds prevail.

    - nb_bundles_per_fixel : np.ndarray (x, y, z)
      For each voxel, it gives the number of bundles associated with each of
      the 5 fixels.

    - nb_bundles_per_voxel : np.ndarray (x, y, z)
      For each voxel, it gives the number of bundles within the voxel. This
      accounts for bundles that might be associated with more than one fixel,
      so no bundle is counted more than once in a voxel.

    If more than one normalization type is given, every output will be computed
    for each normalization. In any case, the normalization type is added after
    the basename of the output, such as fixel_density_maps_voxel-norm.

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

    WARNING: If multiple normalization types are given, along with the
    split_bundles, split_fixels and single_bundle arguments, this script will
    produce a lot of outputs.
"""

import argparse
import nibabel as nib
import numpy as np
import logging
from pathlib import Path

from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             assert_headers_compatible, assert_inputs_exist,
                             add_verbose_arg,
                             assert_output_dirs_exist_and_empty)
from scilpy.tractanalysis.fixel_density import (fixel_density, maps_to_masks)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_peaks',
                   help='Path of the peaks. The peaks are expected to be '
                        'given as unit directions. \nTo get these from fODF '
                        'or SH data, use the script scil_fodf_metrics.py '
                        '\nwith the abs_peaks_and_values option.')

    p.add_argument('--in_bundles', nargs='+', required=True,
                   help='List of paths of the bundles (.trk) to analyze.')

    p.add_argument('--in_bundles_names', nargs='+',
                   help='List of the names of the bundles, in the same order '
                        'as they were given. \nIf this argument is not used, '
                        'the script assumes that the name of the bundle \nis '
                        'its filename without extensions.')

    p.add_argument('--dps_key', default=None, type=str,
                   help='Key to access the data per streamline to use as '
                        'weight when computing the maps, \ninstead of the '
                        'number of streamlines. [%(default)s].')

    p.add_argument('--max_theta', default=45,
                   help='Maximum angle between streamline and peak to be '
                        'associated [%(default)s].')

    g1 = p.add_argument_group(title='Mask parameters')

    g1.add_argument('--abs_thr', default=0, type=float,
                    help='Value of density maps threshold to obtain density '
                         'masks, in number of streamlines \nor streamline '
                         'weighting if --dps_key is given. Any number of '
                         'streamlines \nor weight above this value will '
                         'pass the absolute threshold test [%(default)s].')

    g1.add_argument('--rel_thr', default=0.01, type=float,
                    help='Value of density maps threshold to obtain density '
                         'masks, as a ratio of the normalized density '
                         '\nAny normalized density above '
                         'this value will pass the relative threshold test. '
                         '\nMust be between 0 and 1 [%(default)s].')

    g1.add_argument('--norm', default='voxel', nargs='+', type=str,
                    choices=['fixel', 'voxel', 'none'],
                    help='Way of normalizing the density maps. If fixel, '
                         'will normalize the maps per fixel, \nin each voxel. '
                         'If voxel, will normalize the maps per voxel. \nIf '
                         'none, will not normalize the maps. If multiple '
                         'choices, will save each of them [%(default)s].')

    g2 = p.add_argument_group(title='Output options')

    g2.add_argument('--split_bundles', action='store_true',
                    help='If set, save the density maps and masks for each '
                         'bundle separately \nin addition to the all in one '
                         'version.')

    g2.add_argument('--split_fixels', action='store_true',
                    help='If set, save the density maps and masks for each '
                         'fixel separately \nin addition to the all in one '
                         'version.')

    g2.add_argument('--single_bundle', action='store_true',
                    help='If set, will save the single-fiber single-bundle '
                         'masks as well. \nThese are obtained by '
                         'selecting the voxels where only one bundle is '
                         'present \n(and one fiber/fixel).')

    g2.add_argument('--out_dir', default='fixel_analysis/', type=str,
                    help='Path to the output directory where all the output '
                         'files will be saved. \nCurrent directory by '
                         'default.')

    g2.add_argument('--prefix', default="", type=str,
                    help='Prefix to add to all predetermined output '
                         'filenames. \nWe recommand finishing with an '
                         'underscore for better readability [%(default)s].')

    g2.add_argument('--suffix', default="", type=str,
                    help='Suffix to add to all predetermined output '
                         'filenames. \nWe recommand starting with an '
                         'underscore for better readability [%(default)s].')

    add_overwrite_arg(p)
    add_processes_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Set up saving filename options
    out_dir = args.out_dir
    prefix = args.prefix
    suffix = args.suffix
    if out_dir[-1] != "/":
        out_dir += "/"

    # Set names for saving
    fd_map_name = out_dir + prefix + "fixel_density_map"
    fd_mask_name = out_dir + prefix + "fixel_density_mask"
    vd_map_name = out_dir + prefix + "voxel_density_map"
    vd_mask_name = out_dir + prefix + "voxel_density_mask"
    sb_mask_name = out_dir + prefix + "single_bundle_mask"
    nb_bundles_name = out_dir + prefix + "nb_bundles"

    assert_output_dirs_exist_and_empty(parser, args, out_dir, create_dir=True)
    assert_inputs_exist(parser, [args.in_peaks] + args.in_bundles)
    assert_headers_compatible(parser, [args.in_peaks] + args.in_bundles)

    if args.rel_thr < 0 or args.rel_thr > 1:
        parser.error("Argument rel_thr must be a value between 0 and 1.")

    # Load the data
    logging.info("Loading data.")
    peaks_img = nib.load(args.in_peaks)
    peaks = peaks_img.get_fdata()
    affine = peaks_img.affine

    # Compute NuFo single-fiber from peaks
    if args.single_bundle:
        is_first_peak = np.sum(peaks[..., 0:3], axis=-1) != 0
        is_second_peak = np.sum(peaks[..., 3:6], axis=-1) == 0
        nufo_sf = np.logical_and(is_first_peak, is_second_peak)

    # Extract bundles and names
    bundles = args.in_bundles
    if args.in_bundles_names:  # If names are given
        if len(args.in_bundles_names) != len(bundles):
            parser.error("--in_bundles_names must contain the same number of "
                         "elements as in --in_bundles.")
        bundles_names = args.in_bundles_names
    else:
        logging.info("Extracting bundles names.")
        bundles_names = []
        for bundle in bundles:
            bundles_names.append(Path(bundle).name.split(".")[0])

    # Compute fixel density (FD) maps and masks
    logging.info("Computing fixel density for all bundles.")
    fd_maps_original = fixel_density(peaks, bundles, args.dps_key,
                                     args.max_theta,
                                     nbr_processes=args.nbr_processes)

    for norm in args.norm:
        norm_name = norm + "-norm"
        logging.info("Performing normalization of type {}.".format(norm))
        logging.info("Computing density masks from density maps.")
        fd_masks, fd_maps = maps_to_masks(np.copy(fd_maps_original),
                                          args.abs_thr, args.rel_thr, norm,
                                          len(bundles))

        logging.info("Computing additional derivatives.")
        # Compute number of bundles per fixel
        nb_bundles_per_fixel = np.sum(fd_masks, axis=-1)
        # Compute voxel density (VD) maps
        vd_maps = np.sum(fd_maps, axis=-2)
        # Compute voxel density (VD) masks
        # Since a bundle can be present twice in a single voxel by being
        # associated with more than one fixel, we count the presence of a
        # bundle if > 0.
        vd_masks = np.where(np.sum(fd_masks, axis=-2) > 0, 1, 0)
        # Compute number of bundles per voxel by taking the sum of the mask
        nb_bundles_per_voxel = np.sum(vd_masks, axis=-1)

        # Save all results
        logging.info("Saving all results.")
        for i, bundle_n in enumerate(bundles_names):
            if args.split_bundles:  # Save the maps and masks for each bundle
                nib.save(nib.Nifti1Image(fd_maps[..., i], affine),
                         "{}_{}_{}{}.nii.gz".format(fd_map_name, norm_name,
                                                    bundle_n, suffix))
                nib.save(nib.Nifti1Image(fd_masks[..., i], affine),
                         "{}_{}_{}{}.nii.gz".format(fd_mask_name, norm_name,
                                                    bundle_n, suffix))
                if norm != "fixel":  # If fixel, voxel maps mean nothing
                    nib.save(nib.Nifti1Image(vd_maps[..., i], affine),
                             "{}_{}_{}{}.nii.gz".format(vd_map_name, norm_name,
                                                        bundle_n, suffix))
                bundle_mask = vd_masks[..., i].astype(np.uint8)
                nib.save(nib.Nifti1Image(bundle_mask, affine),
                         "{}_{}_{}{}.nii.gz".format(vd_mask_name, norm_name,
                                                    bundle_n, suffix))

            if args.single_bundle:
                # Single-fiber single-bundle voxels
                one_bundle_per_voxel = nb_bundles_per_voxel == 1
                # Making sure we also have single-fiber voxels only
                one_bundle_per_voxel *= nufo_sf
                # Save a single-fiber single-bundle mask for the whole WM
                nib.save(nib.Nifti1Image(one_bundle_per_voxel.astype(np.uint8),
                                         affine),
                         "{}_{}_WM{}.nii.gz".format(sb_mask_name, norm_name,
                                                    suffix))
                # Save a single-fiber single-bundle mask for each bundle
                bundle_mask = fd_masks[..., 0, i] * one_bundle_per_voxel
                nib.save(nib.Nifti1Image(bundle_mask.astype(np.uint8), affine),
                         "{}_{}_{}{}.nii.gz".format(sb_mask_name, norm_name,
                                                    bundle_n, suffix))

        if args.split_fixels:  # Save the maps and masks for each fixel
            for i in range(5):
                nib.save(nib.Nifti1Image(fd_maps[..., i, :], affine),
                         "{}_{}_f{}{}.nii.gz".format(fd_map_name, norm_name,
                                                     i + 1, suffix))
                nib.save(nib.Nifti1Image(fd_masks[..., i, :], affine),
                         "{}_{}_f{}{}.nii.gz".format(fd_mask_name, norm_name,
                                                     i + 1, suffix))
                if norm != "fixel":  # If fixel, voxel maps mean nothing
                    nib.save(nib.Nifti1Image(vd_maps[..., i, :], affine),
                             "{}_{}_f{}{}.nii.gz".format(vd_map_name,
                                                         norm_name, i + 1,
                                                         suffix))
                bundle_mask = vd_masks[..., i, :].astype(np.uint8)
                nib.save(nib.Nifti1Image(bundle_mask, affine),
                         "{}_{}_f{}{}.nii.gz".format(vd_mask_name, norm_name,
                                                     i + 1, suffix))

        # Save full fixel density maps, all fixels and bundles combined
        nib.save(nib.Nifti1Image(fd_maps, affine),
                 "{}s_{}{}.nii.gz".format(fd_map_name, norm_name, suffix))

        # Save full fixel density masks, all fixels and bundles combined
        nib.save(nib.Nifti1Image(fd_masks, affine),
                 "{}s_{}{}.nii.gz".format(fd_mask_name, norm_name, suffix))

        # Save full voxel density maps and masks
        if norm != "fixel":  # If fixel, voxel maps mean nothing
            nib.save(nib.Nifti1Image(vd_maps, affine),
                     "{}s_{}{}.nii.gz".format(vd_map_name, norm_name, suffix))
        nib.save(nib.Nifti1Image(vd_masks.astype(np.uint8), affine),
                 "{}s_{}{}.nii.gz".format(vd_mask_name, norm_name, suffix))

        # Save number of bundles per fixel and per voxel
        nib.save(nib.Nifti1Image(nb_bundles_per_fixel.astype(np.uint8),
                                 affine),
                 "{}_per_fixel_{}{}.nii.gz".format(nb_bundles_name, norm_name,
                                                   suffix))
        nib.save(nib.Nifti1Image(nb_bundles_per_voxel.astype(np.uint8),
                                 affine),
                 "{}_per_voxel_{}{}.nii.gz".format(nb_bundles_name, norm_name,
                                                   suffix))

    # Save bundles lookup table to know the order of the bundles
    bundles_idx = np.arange(0, len(bundles_names), 1)
    lookup_table = np.array([bundles_names, bundles_idx])
    np.savetxt("{}{}bundles_LUT{}.txt".format(out_dir, prefix, suffix),
               lookup_table, fmt='%s')


if __name__ == "__main__":
    main()
