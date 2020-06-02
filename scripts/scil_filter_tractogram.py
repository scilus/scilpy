#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
Now supports sequential filtering condition and mixed filtering object.
For example, --atlas_roi ROI_NAME ID MODE CRITERIA
- ROI_NAME is the filename of a Nifti
- ID is the integer value in the atlas
- MODE must be one of these values: 'any', 'either_end', 'both_ends'
- CRITERIA must be one of these values: ['include', 'exclude']

Multiple filtering tuples can be used and options mixed.
A logical AND is the only behavior available. All theses filtering
conditions will be sequentially applied.
"""

import argparse
import json
import logging
import os

from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_label, get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             read_info_from_mb_bdo)
from scilpy.segment.streamlines import (filter_cuboid, filter_ellipsoid,
                                        filter_grid_roi)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file.')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file.')

    p.add_argument('--drawn_roi', nargs=3, action='append',
                   metavar=('ROI_NAME', 'MODE', 'CRITERIA'),
                   help='Filename of a hand drawn ROI (.nii or .nii.gz).')
    p.add_argument('--atlas_roi', nargs=4, action='append',
                   metavar=('ROI_NAME', 'ID', 'MODE', 'CRITERIA'),
                   help='Filename of an atlas (.nii or .nii.gz).')
    p.add_argument('--bdo', nargs=3, action='append',
                   metavar=('BDO_NAME', 'MODE', 'CRITERIA'),
                   help='Filename of a bounding box (bdo) file from MI-Brain.')

    p.add_argument('--x_plane', nargs=3, action='append',
                   metavar=('PLANE', 'MODE', 'CRITERIA'),
                   help='Slice number in X, in voxel space.')
    p.add_argument('--y_plane', nargs=3, action='append',
                   metavar=('PLANE', 'MODE', 'CRITERIA'),
                   help='Slice number in Y, in voxel space.')
    p.add_argument('--z_plane', nargs=3, action='append',
                   metavar=('PLANE', 'MODE', 'CRITERIA'),
                   help='Slice number in Z, in voxel space.')
    p.add_argument('--filtering_list',
                   help='Text file containing one rule per line\n'
                   '(i.e. drawn_roi mask.nii.gz both_ends include).')
    p.add_argument('--no_empty', action='store_true',
                   help='Do not write file if there is no streamline.')
    p.add_argument('--display_counts', action='store_true',
                   help='Print streamline count before and after filtering')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_json_args(p)

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
            roi_opt_list.append(roi_opt.strip().split())

    for roi_opt in roi_opt_list:
        if roi_opt[0] == 'atlas_roi':
            filter_type, filter_arg, _, filter_mode, filter_criteria = roi_opt
        else:
            filter_type, filter_arg, filter_mode, filter_criteria = roi_opt
        if filter_type not in ['x_plane', 'y_plane', 'z_plane']:
            if not os.path.isfile(filter_arg):
                parser.error('{} does not exist'.format(filter_arg))
        if filter_mode not in ['any', 'either_end', 'both_ends']:
            parser.error('{} is not a valid option for filter_mode'.format(
                filter_mode))
        if filter_criteria not in ['include', 'exclude']:
            parser.error('{} is not a valid option for filter_criteria'.format(
                filter_criteria))

    return roi_opt_list, only_filtering_list


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    roi_opt_list, only_filtering_list = prepare_filtering_list(parser, args)
    o_dict = {}

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    # Streamline count before filtering
    o_dict['streamline_count_before_filtering'] = len(sft.streamlines)

    for i, roi_opt in enumerate(roi_opt_list):
        curr_dict = {}
        # Atlas needs an extra argument (value in the LUT)
        if roi_opt[0] == 'atlas_roi':
            filter_type, filter_arg, filter_arg_2, \
                filter_mode, filter_criteria = roi_opt
        else:
            filter_type, filter_arg, filter_mode, filter_criteria = roi_opt

        curr_dict['filename'] = os.path.abspath(filter_arg)
        curr_dict['type'] = filter_type
        curr_dict['mode'] = filter_mode
        curr_dict['criteria'] = filter_criteria

        is_exclude = False if filter_criteria == 'include' else True

        if filter_type == 'drawn_roi' or filter_type == 'atlas_roi':
            img = nib.load(filter_arg)
            if not is_header_compatible(img, sft):
                parser.error('Headers from the tractogram and the mask are '
                             'not compatible.')
            if filter_type == 'drawn_roi':
                mask = get_data_as_mask(img)
            else:
                atlas = get_data_as_label(img)
                mask = np.zeros(atlas.shape, dtype=np.uint16)
                mask[atlas == int(filter_arg_2)] = 1
            filtered_sft, indexes = filter_grid_roi(sft, mask,
                                                    filter_mode, is_exclude)

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

            filtered_sft, indexes = filter_grid_roi(sft, mask,
                                                    filter_mode, is_exclude)

        elif filter_type == 'bdo':
            geometry, radius, center = read_info_from_mb_bdo(filter_arg)
            if geometry == 'Ellipsoid':
                filtered_sft, indexes = filter_ellipsoid(sft,
                                                         radius, center,
                                                         filter_mode, is_exclude)
            elif geometry == 'Cuboid':
                filtered_sft, indexes = filter_cuboid(sft,
                                                      radius, center,
                                                      filter_mode, is_exclude)

        logging.debug('The filtering options {0} resulted in '
                      '{1} streamlines'.format(roi_opt, len(filtered_sft)))

        sft = filtered_sft

        if only_filtering_list:
            filtering_Name = 'Filter_' + str(i)
            curr_dict['streamline_count_after_filtering'] = len(sft.streamlines)
            o_dict[filtering_Name] = curr_dict

    # Streamline count after filtering
    o_dict['streamline_count_final_filtering'] = len(sft.streamlines)
    if args.display_counts:
        print(json.dumps(o_dict, indent=args.indent))

    if not filtered_sft:
        if args.no_empty:
            logging.debug("The file {} won't be written (0 streamline)".format(
                args.out_tractogram))

            return

        logging.debug('The file {} contains 0 streamline'.format(
            args.out_tractogram))

    save_tractogram(sft, args.out_tractogram)


if __name__ == "__main__":
    main()
