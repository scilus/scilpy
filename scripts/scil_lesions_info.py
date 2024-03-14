#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will output informations about lesion load in bundle(s).
The input can either be streamlines, binary bundle map, or a bundle voxel
label map.

To be considered a valid lesion, the lesion volume must be at least
min_lesion_vol mm3. This avoid the detection of thousand of single voxel
lesions if an automatic lesion segmentation tool is used.

Formerly: scil_analyse_lesions_load.py
"""

import argparse
import json
import logging
import os

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from scilpy.image.labels import get_data_as_labels
from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             add_json_args, assert_outputs_exist,
                             add_verbose_arg, add_reference_arg,
                             assert_headers_compatible)
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.metrics_tools import compute_lesion_stats


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_lesion',
                   help='Binary mask of the lesion(s) (.nii.gz).')
    p.add_argument('out_json',
                   help='Output file for lesion information (.json).')
    p1 = p.add_mutually_exclusive_group()
    p1.add_argument('--bundle',
                    help='Path of the bundle file (.trk).')
    p1.add_argument('--bundle_mask',
                    help='Path of the bundle binary mask (.nii.gz).')
    p1.add_argument('--bundle_labels_map',
                    help='Path of the bundle labels map (.nii.gz).')

    p.add_argument('--min_lesion_vol', type=float, default=7,
                   help='Minimum lesion volume in mm3 [%(default)s].')
    p.add_argument('--out_lesion_atlas', metavar='FILE',
                   help='Save the labelized lesion(s) map (.nii.gz).')
    p.add_argument('--out_lesion_stats', metavar='FILE',
                   help='Save the lesion-wise volume measure (.json).')
    p.add_argument('--out_streamlines_stats', metavar='FILE',
                   help='Save the lesion-wise streamline count (.json).')

    add_json_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if (not args.bundle) and (not args.bundle_mask) \
            and (not args.bundle_labels_map):
        parser.error('One of the option --bundle or --map must be used')

    assert_inputs_exist(parser, args.in_lesion,
                        optional=[args.bundle, args.bundle_mask,
                                  args.bundle_labels_map, args.reference])
    assert_outputs_exist(parser, args, args.out_json,
                         optional=[args.out_lesion_stats,
                                   args.out_streamlines_stats])
    assert_headers_compatible(parser, args.in_lesion,
                              optional=[args.bundle, args.bundle_mask,
                                        args.bundle_labels_map],
                              reference=args.reference)

    lesion_img = nib.load(args.in_lesion)
    lesion_data = get_data_as_mask(lesion_img, dtype=bool)

    if args.bundle:
        bundle_name, _ = split_name_with_nii(os.path.basename(args.bundle))
        sft = load_tractogram_with_reference(parser, args, args.bundle)
        sft.to_vox()
        sft.to_corner()
        streamlines = sft.get_streamlines_copy()
        map_data = compute_tract_counts_map(streamlines,
                                            lesion_data.shape)
        map_data[map_data > 0] = 1
    elif args.bundle_mask:
        bundle_name, _ = split_name_with_nii(
            os.path.basename(args.bundle_mask))
        map_img = nib.load(args.bundle_mask)
        map_data = get_data_as_mask(map_img)
    else:
        bundle_name, _ = split_name_with_nii(os.path.basename(
            args.bundle_labels_map))
        map_img = nib.load(args.bundle_labels_map)
        map_data = get_data_as_labels(map_img)

    is_single_label = args.bundle_labels_map is None
    voxel_sizes = lesion_img.header.get_zooms()[0:3]
    lesion_atlas, _ = ndi.label(lesion_data)

    lesion_load_dict = compute_lesion_stats(
        map_data, lesion_atlas, single_label=is_single_label,
        voxel_sizes=voxel_sizes, min_lesion_vol=args.min_lesion_vol)

    if args.out_lesion_atlas:
        lesion_atlas *= map_data.astype(bool)
        nib.save(nib.Nifti1Image(lesion_atlas, lesion_img.affine),
                 args.out_lesion_atlas)

    volume_dict = {bundle_name: lesion_load_dict}
    with open(args.out_json, 'w') as outfile:
        json.dump(volume_dict, outfile,
                  sort_keys=args.sort_keys, indent=args.indent)

    if args.out_streamlines_stats or args.out_lesion_stats:
        lesion_dict = {}
        for lesion in np.unique(lesion_atlas)[1:]:
            curr_vol = np.count_nonzero(lesion_atlas[lesion_atlas == lesion]) \
                * np.prod(voxel_sizes)
            if curr_vol >= args.min_lesion_vol:
                key = str(lesion).zfill(4)
                lesion_dict[key] = {'volume': curr_vol}
                if args.bundle:
                    tmp = np.zeros(lesion_atlas.shape)
                    tmp[lesion_atlas == lesion] = 1
                    new_sft, _ = filter_grid_roi(sft, tmp, 'any', False)
                    lesion_dict[key]['strs_count'] = len(new_sft)

        lesion_vol_dict = {bundle_name: {}}
        streamlines_count_dict = {bundle_name: {'streamlines_count': {}}}
        for key in lesion_dict.keys():
            lesion_vol_dict[bundle_name][key] = lesion_dict[key]['volume']
            if args.bundle:
                streamlines_count_dict[bundle_name]['streamlines_count'][key] = \
                    lesion_dict[key]['strs_count']

        if args.out_lesion_stats:
            with open(args.out_lesion_stats, 'w') as outfile:
                json.dump(lesion_vol_dict, outfile,
                          sort_keys=args.sort_keys, indent=args.indent)
        if args.out_streamlines_stats:
            with open(args.out_streamlines_stats, 'w') as outfile:
                json.dump(streamlines_count_dict, outfile,
                          sort_keys=args.sort_keys, indent=args.indent)


if __name__ == "__main__":
    main()
