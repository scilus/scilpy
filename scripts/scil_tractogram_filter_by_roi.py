#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
Now supports sequential filtering condition and mixed filtering object.
For example, --atlas_roi ROI_NAME ID MODE CRITERIA DISTANCE
- ROI_NAME is the filename of a Nifti
- ID is one or multiple integer values in the atlas. If multiple values,
    ID needs to be between quotes.
    Example: "1:6 9 10:15" will use values between 1 and 6 and values
                           between 10 and 15 included as well as value 9.
- MODE must be one of these values: ['any', 'all', 'either_end', 'both_ends']
- CRITERIA must be one of these values: ['include', 'exclude']
- DISTANCE must be a int and is optional

If any meant any part of the streamline must be in the mask, all means that
all part of the streamline must be in the mask.

When used with exclude, it means that a streamline entirely in the mask will
be excluded. Using all it with x/y/z plane works but makes very little sense.

In terms of nifti mask, --drawn_roi MASK.nii.gz all include is
equivalent to --drawn_roi INVERSE_MASK.nii.gz any exclude
For example, this allows to find out all streamlines entirely in the WM in
one command (without manually inverting the mask first) or
to remove any streamlines staying in GM without getting out.

Multiple filtering tuples can be used and options mixed.
A logical AND is the only behavior available. All theses filtering
conditions will be sequentially applied.

WARNING: DISTANCE is optional and it should be used carefully with large
voxel size (e.g > 2.5mm). The value is in voxel for ROIs and in mm for
bounding box. Anisotropic data will affect each direction differently

Formerly: scil_filter_tractogram.py
"""

import argparse
import json
import logging
import os
from copy import deepcopy

from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.io.image import (get_data_as_mask,
                             merge_labels_into_mask)
from scilpy.image.labels import get_data_as_labels
from scilpy.io.streamlines import load_tractogram_with_reference, \
    save_tractogram
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             read_info_from_mb_bdo)
from scilpy.segment.streamlines import (filter_cuboid, filter_ellipsoid,
                                        filter_grid_roi)


MODES = ['any', 'all', 'either_end', 'both_ends']
CRITERIA = ['include', 'exclude']


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file.')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file.')

    p.add_argument('--drawn_roi', nargs='+', action='append',
                   help="ROI_NAME MODE CRITERIA DISTANCE "
                        "(distance in voxel is optional)\n"
                        "Filename of a hand drawn ROI (.nii or .nii.gz).")
    p.add_argument('--atlas_roi', nargs='+', action='append',
                   help="ROI_NAME ID MODE CRITERIA DISTANCE "
                        "(distance in voxel is optional)\n"
                        "Filename of an atlas (.nii or .nii.gz).")
    p.add_argument('--bdo', nargs='+', action='append',
                   help="BDO_NAME MODE CRITERIA DISTANCE "
                        "(distance in mm is optional)\n"
                        "Filename of a bounding box (bdo) file from MI-Brain.")

    p.add_argument('--x_plane', nargs='+', action='append',
                   help="PLANE MODE CRITERIA DISTANCE "
                        "(distance in voxel is optional)\n"
                        "Slice number in X, in voxel space.")
    p.add_argument('--y_plane', nargs='+', action='append',
                   help="PLANE MODE CRITERIA DISTANCE "
                        "(distance in voxel is optional)\n"
                        "Slice number in Y, in voxel space.")
    p.add_argument('--z_plane', nargs='+', action='append',
                   help="PLANE MODE CRITERIA DISTANCE "
                        "(distance in voxel is optional)\n"
                        "Slice number in Z, in voxel space.")
    p.add_argument('--filtering_list',
                   help='Text file containing one rule per line\n'
                   '(i.e. drawn_roi mask.nii.gz both_ends include 1).')

    p.add_argument('--overwrite_distance', nargs='+', action='append',
                   help='MODE CRITERIA DISTANCE (distance in voxel for ROIs '
                        'and in mm for bounding box).\n'
                        'If set, it will overwrite the distance associated to '
                        'a specific mode/criteria.')

    p.add_argument('--extract_masks_atlas_roi', action='store_true',
                   help='Extract atlas roi masks.')
    p.add_argument('--no_empty', action='store_true',
                   help='Do not write file if there is no streamline.')
    p.add_argument('--display_counts', action='store_true',
                   help='Print streamline count before and after filtering')
    p.add_argument('--save_rejected', metavar='FILENAME',
                   help='Save rejected streamlines to output tractogram.')

    add_json_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def prepare_filtering_list(parser, args):
    roi_opt_list = []
    only_filtering_list = True

    if args.drawn_roi:
        only_filtering_list = False
        for roi_opt in args.drawn_roi:
            roi_opt_list.append(['drawn_roi'] + roi_opt)
    if args.atlas_roi:
        only_filtering_list = False
        for roi_opt in args.atlas_roi:
            roi_opt_list.append(['atlas_roi'] + roi_opt)
    if args.bdo:
        only_filtering_list = False
        for roi_opt in args.bdo:
            roi_opt_list.append(['bdo'] + roi_opt)
    if args.x_plane:
        only_filtering_list = False
        for roi_opt in args.x_plane:
            roi_opt_list.append(['x_plane'] + roi_opt)
    if args.y_plane:
        only_filtering_list = False
        for roi_opt in args.y_plane:
            roi_opt_list.append(['y_plane'] + roi_opt)
    if args.z_plane:
        only_filtering_list = False
        for roi_opt in args.z_plane:
            roi_opt_list.append(['z_plane'] + roi_opt)
    if args.filtering_list:
        with open(args.filtering_list) as txt:
            content = txt.readlines()
        for roi_opt in content:
            if "\"" in roi_opt:
                tmp_opt = [i.strip() for i in roi_opt.strip().split("\"")]
                roi_opt_list.append(
                    tmp_opt[0].split() + [tmp_opt[1]] + tmp_opt[2].split())
            else:
                roi_opt_list.append(roi_opt.strip().split())

    if (len(roi_opt_list[-1]) < 4 or len(roi_opt_list[-1]) > 5) and \
            roi_opt_list[-1][0] != 'atlas_roi':
        logging.error("Please specify 3 or 4 values "
                      "for {} filtering.".format(roi_opt_list[-1][0]))
    elif (len(roi_opt_list[-1]) < 5 or len(roi_opt_list[-1]) > 6) and \
            roi_opt_list[-1][0] == 'atlas_roi':
        logging.error("Please specify 4 or 5 values"
                      " for {} filtering.".format(roi_opt_list[-1][0]))

    filter_distance = 0
    for index, roi_opt in enumerate(roi_opt_list):
        if roi_opt[0] == 'atlas_roi':
            if len(roi_opt) == 5:
                filter_type, filter_arg, _, filter_mode, filter_criteria = \
                                                                    roi_opt
                roi_opt_list[index].append(0)
            else:
                filter_type, filter_arg, _, filter_mode, filter_criteria, \
                    filter_distance = roi_opt
        elif len(roi_opt) == 4:
            filter_type, filter_arg, filter_mode, filter_criteria = roi_opt
            roi_opt_list[index].append(0)
        else:
            filter_type, filter_arg, filter_mode, filter_criteria, \
                filter_distance = roi_opt

        if filter_type not in ['x_plane', 'y_plane', 'z_plane']:
            if not os.path.isfile(filter_arg):
                parser.error('{} does not exist'.format(filter_arg))
        if filter_mode not in ['any', 'all', 'either_end', 'both_ends']:
            parser.error('{} is not a valid option for filter_mode'.format(
                filter_mode))
        if filter_criteria not in ['include', 'exclude']:
            parser.error('{} is not a valid option for filter_criteria'.format(
                filter_criteria))

        if int(filter_distance) < 0:
            parser.error("Distance should be positive. "
                         "{} is not a valid option.".format(filter_distance))

    return roi_opt_list, only_filtering_list


def check_overwrite_distance(parser, args):
    dict_distance = {}
    if args.overwrite_distance:
        for distance in args.overwrite_distance:
            if len(distance) != 3:
                parser.error('overwrite_distance is not well formated.\n'
                             'It should be MODE CRITERIA DISTANCE.')
            elif '-'.join([distance[0], distance[1]]) in dict_distance:
                parser.error('Overwrite distance dictionnary MODE '
                             '"{}" has been set multiple times.'.format(
                                                                distance[0]))
            elif distance[0] in MODES and distance[1] in CRITERIA:
                curr_key = '-'.join([distance[0], distance[1]])
                dict_distance[curr_key] = distance[2]
            else:
                curr_key = '-'.join([distance[0], distance[1]])
                parser.error('Overwrite distance dictionnary MODE-CRITERIA '
                             '"{}" does not exist.'.format(curr_key))
    else:
        return dict_distance

    return dict_distance


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    overwrite_distance = check_overwrite_distance(parser, args)

    # Todo. Prepare now the names of other files (ex, ROI) and verify if
    #  exist and compatible.
    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram, args.save_rejected)

    if overwrite_distance:
        logging.info('Overwrite distance dictionnary {}'.format(
                                                        overwrite_distance))

    roi_opt_list, only_filtering_list = prepare_filtering_list(parser, args)
    o_dict = {}

    logging.info("Loading the tractogram...")
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    if args.save_rejected:
        initial_sft = deepcopy(sft)

    # Streamline count before filtering
    o_dict['streamline_count_before_filtering'] = len(sft.streamlines)

    atlas_roi_item = 0

    total_kept_ids = np.arange(len(sft.streamlines))
    for i, roi_opt in enumerate(roi_opt_list):
        logging.info("Preparing filtering from option: {}".format(roi_opt))
        curr_dict = {}
        # Atlas needs an extra argument (value in the LUT)
        if roi_opt[0] == 'atlas_roi':
            filter_type, filter_arg, filter_arg_2, \
                filter_mode, filter_criteria, filter_distance = roi_opt
        else:
            filter_type, filter_arg, filter_mode, filter_criteria, \
                filter_distance = roi_opt

        curr_dict['filename'] = os.path.abspath(filter_arg)
        curr_dict['type'] = filter_type
        curr_dict['mode'] = filter_mode
        curr_dict['criteria'] = filter_criteria

        key_distance = '-'.join([curr_dict['mode'], curr_dict['criteria']])
        if key_distance in overwrite_distance:
            curr_dict['distance'] = overwrite_distance[key_distance]
        else:
            curr_dict['distance'] = filter_distance

        try:
            filter_distance = int(curr_dict['distance'])
        except ValueError:
            parser.error('Distance filter {} should is not an integer.'.format(
                                                        curr_dict['distance']))

        is_exclude = False if filter_criteria == 'include' else True

        if filter_type == 'drawn_roi' or filter_type == 'atlas_roi':
            img = nib.load(filter_arg)
            if not is_header_compatible(img, sft):
                parser.error('Headers from the tractogram and the mask are '
                             'not compatible.')
            if filter_type == 'drawn_roi':
                mask = get_data_as_mask(img)
            else:
                atlas = get_data_as_labels(img)
                mask = merge_labels_into_mask(atlas, filter_arg_2)

                if args.extract_masks_atlas_roi:
                    atlas_roi_item = atlas_roi_item + 1
                    nib.Nifti1Image(mask.astype(np.uint16),
                                    img.affine).to_filename(
                                        'mask_atlas_roi_{}.nii.gz'.format(
                                            str(atlas_roi_item)))

            filtered_sft, kept_ids = filter_grid_roi(sft, mask,
                                                     filter_mode, is_exclude,
                                                     filter_distance)

        # For every case, the input number must be greater or equal to 0 and
        # below the dimension, since this is a voxel space operation
        elif filter_type in ['x_plane', 'y_plane', 'z_plane']:
            filter_arg = int(filter_arg)
            _, dim, _, _ = sft.space_attributes
            mask = np.zeros(dim, dtype=np.int16)
            error_msg = None
            if filter_type == 'x_plane':
                if 0 <= filter_arg < dim[0]:
                    mask[filter_arg, :, :] = 1
                else:
                    error_msg = 'X plane ' + str(filter_arg)

            elif filter_type == 'y_plane':
                if 0 <= filter_arg < dim[1]:
                    mask[:, filter_arg, :] = 1
                else:
                    error_msg = 'Y plane ' + str(filter_arg)

            elif filter_type == 'z_plane':
                if 0 <= filter_arg < dim[2]:
                    mask[:, :, filter_arg] = 1
                else:
                    error_msg = 'Z plane ' + str(filter_arg)

            if error_msg:
                parser.error('{} is not valid according to the '
                             'tractogram header.'.format(error_msg))

            filtered_sft, kept_ids = filter_grid_roi(sft, mask,
                                                     filter_mode, is_exclude,
                                                     filter_distance)

        elif filter_type == 'bdo':
            geometry, radius, center = read_info_from_mb_bdo(filter_arg)

            if filter_distance != 0:
                radius += filter_distance * sft.space_attributes[2]

            if geometry == 'Ellipsoid':
                filtered_sft, kept_ids = filter_ellipsoid(
                    sft, radius, center, filter_mode, is_exclude)
            elif geometry == 'Cuboid':
                filtered_sft, kept_ids = filter_cuboid(
                    sft, radius, center, filter_mode, is_exclude)
        else:
            raise ValueError("Unexpected filter type.")

        logging.info('The filtering options {0} resulted in '
                     '{1} streamlines'.format(roi_opt, len(filtered_sft)))

        sft = filtered_sft

        if only_filtering_list:
            filtering_Name = 'Filter_' + str(i)
            curr_dict['streamline_count_after_filtering'] = len(
                sft.streamlines)
            o_dict[filtering_Name] = curr_dict

        total_kept_ids = total_kept_ids[kept_ids]

    # Streamline count after filtering
    o_dict['streamline_count_final_filtering'] = len(sft.streamlines)
    if args.display_counts:
        print(json.dumps(o_dict, indent=args.indent))

    save_tractogram(sft, args.out_tractogram,
                    args.no_empty)

    if args.save_rejected:
        rejected_ids = np.setdiff1d(np.arange(len(initial_sft.streamlines)),
                                    total_kept_ids)

        sft = initial_sft[rejected_ids]
        save_tractogram(sft, args.save_rejected,
                        args.no_empty)


if __name__ == "__main__":
    main()
