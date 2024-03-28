#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
Filtering of a tractogram based on any combination of conditions involving a
ROI (ex: keep streamlines whose endoints are inside the ROI, exclude
streamlines not entirely included in a ROI, etc.)

See also:
    - scil_tractogram_detect_loops.py
    - scil_tractogram_filter_by_anatomy.py
        (Can reject streamlines with endpoints in the WM or the CSF based on
        labels)
    - scil_tractogram_filter_by_length.py
    - scil_tractogram_filter_by_orientation.py

Condition
---------
For every type of ROI, two or three values are expected: MODE CRITERIA DISTANCE
(DISTANCE is always optional)
- MODE must be one of these values: ['any', 'all', 'either_end', 'both_ends']
    - any: any part of the streamline must be in the mask
    - all: all parts of the streamline must be in the mask.
    - either_end: at least one end of the streamline must be in the mask.
    - both_ends:  both ends of the streamline must be in the mask.
- CRITERIA must be one of these values: ['include', 'exclude']
    - Include: If condition from MODE is respected, streamline is included.
    - Exlucde: If condition from MODE is respected, streamline is excluded.
- DISTANCE must be an int and is optional.


Type of ROI
-----------
- Drawn ROI: Directly loaded from a binary file.
- Atlas ROI: Selected label from an atlas.
    - ID is one or multiple integer values in the atlas. If multiple values,
        ID needs to be between quotes.
        Example: "1:6 9 10:15" will use values between 1 and 6 and values
                               between 10 and 15 included as well as value 9.
- BDO: The ROI is the interior of a bounding box.
- Planes: The ROI is the equivalent of a one-voxel plane.
    * Using mode 'all' with x/y/z plane works but makes very little sense.

Note: `--drawn_roi MASK.nii.gz all include` is equivalent to
      `--drawn_roi INVERSE_MASK.nii.gz any exclude`

For example, this allows to find out all streamlines entirely in the WM in one
command (without manually inverting the mask first) or to remove any streamline
staying in the GM without getting out.

Supports multiple filtering conditions
--------------------------------------
Multiple filtering conditions can be used, with varied ROI types if necessary.
Combining two conditions is equivalent to a logical AND between the conditions.
Order of application does not matter for the final result, but may change the
intermediate files, if any.

Distance management
-------------------
DISTANCE is optional, and it should be used carefully with large voxel size
(e.g > 2.5mm). The value is in voxel for ROIs and in mm for bounding boxes.
Anisotropic data will affect each direction differently.
    When using --overwrite_distance, any filtering option with given criteria
will have its DISTANCE value replaced.

Formerly: scil_filter_tractogram.py
"""

import argparse
import glob
import json
import logging
import os
from copy import deepcopy

import nibabel as nib
import numpy as np

from scilpy.io.image import (get_data_as_mask,
                             merge_labels_into_mask)
from scilpy.image.labels import get_data_as_labels
from scilpy.io.streamlines import (load_tractogram_with_reference,
                                   save_tractogram)
from scilpy.io.utils import (add_json_args, add_overwrite_arg,
                             add_reference_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             read_info_from_mb_bdo, assert_headers_compatible)
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

    p.add_argument('--drawn_roi', nargs='+', action='append', default=[],
                   help="ROI_NAME MODE CRITERIA DISTANCE "
                        "(distance in voxel is optional)\n"
                        "Filename of a hand drawn ROI (.nii or .nii.gz).")
    p.add_argument('--atlas_roi', nargs='+', action='append', default=[],
                   help="ATLAS_NAME ID MODE CRITERIA DISTANCE "
                        "(distance in voxel is optional)\n"
                        "Filename of an atlas (.nii or .nii.gz).")
    p.add_argument('--bdo', nargs='+', action='append', default=[],
                   help="BDO_NAME MODE CRITERIA DISTANCE "
                        "(distance in mm is optional)\n"
                        "Filename of a bounding box (bdo) file from MI-Brain.")

    p.add_argument('--x_plane', nargs='+', action='append', default=[],
                   help="PLANE MODE CRITERIA DISTANCE "
                        "(distance in voxel is optional)\n"
                        "Slice number in X, in voxel space.")
    p.add_argument('--y_plane', nargs='+', action='append', default=[],
                   help="PLANE MODE CRITERIA DISTANCE "
                        "(distance in voxel is optional)\n"
                        "Slice number in Y, in voxel space.")
    p.add_argument('--z_plane', nargs='+', action='append', default=[],
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

    p.add_argument('--extract_masks_atlas_roi',
                   help='Extract atlas roi masks. Provided value is the '
                        'prefix, \nex: my_path/atlas_roi_. Whole filename '
                        'will be my_path/atlas_roi_{id}.nii.gz')
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


def _convert_filering_list_to_roi_args(parser, args):
    with open(args.filtering_list) as txt:
        content = txt.readlines()
    for roi_opt in content:
        if "\"" in roi_opt:
            # Hein?? À tester
            tmp_opt = [i.strip() for i in roi_opt.strip().split("\"")]
            tmp_opt = [tmp_opt[0].split() + [tmp_opt[1]] + tmp_opt[2].split()]
        else:
            tmp_opt = roi_opt.strip().split()

        if tmp_opt[0] == 'drawn_roi':
            args.drawn_roi.append(tmp_opt[1:])
        elif tmp_opt[0] == 'atlas_roi':
            args.atlas_roi.append(tmp_opt[1:])
        elif tmp_opt[0] == 'bdo':
            args.bdo.append(tmp_opt[1:])
        elif tmp_opt[0] == 'x_plane':
            args.x_plane.append(tmp_opt[1:])
        elif tmp_opt[0] == 'y_plane':
            args.y_plane.append(tmp_opt[1:])
        elif tmp_opt[0] == 'z_plane':
            args.z_plane.append(tmp_opt[1:])
        else:
            parser.error("Filtering list option {} not understood."
                         .format(tmp_opt[0]))

    return args


def _read_and_check_overwrite_distance(parser, args):
    dict_distance = {}
    if args.overwrite_distance:
        for distance in args.overwrite_distance:
            if len(distance) != 3:
                parser.error('overwrite_distance is not well formated.\n'
                             'It should be MODE CRITERIA DISTANCE.')
            elif '-'.join([distance[0], distance[1]]) in dict_distance:
                parser.error('Overwrite distance dictionnary MODE {}, '
                             'CRITERIA {} has been set multiple times.'
                             .format(distance[0], distance[1]))
            elif distance[0] in MODES and distance[1] in CRITERIA:
                if not distance[2].is_integer():
                    parser.error(
                        "Distance must be an int. {} is not a valid option.")
                if distance[2] < 0:
                    parser.error(
                        "Distance should be positive. {} is not a valid "
                        "option.".format(distance[2]))

                curr_key = '-'.join([distance[0], distance[1]])
                dict_distance[curr_key] = distance[2]
            else:
                curr_key = '-'.join([distance[0], distance[1]])
                parser.error('Overwrite distance dictionnary MODE-CRITERIA '
                             '"{}" does not exist.'.format(curr_key))
        logging.info('Overwrite distance dictionnary {}'
                     .format(dict_distance))
    return dict_distance


def _prepare_filtering_criteria(
        parser, drawn_roi, atlas_roi, bdo, x_plane, y_plane, z_plane, dim):
    """

    Returns
    -------
    roi_opt_list: list
        List of criteria (roi_type MODE CRITERIA DISTANCE)
    """
    def _check_values(_mode, _criteria, _distance):
        if _mode not in ['any', 'all', 'either_end', 'both_ends']:
            parser.error('{} is not a valid option for filter_mode'
                         .format(_mode))
        if _criteria not in ['include', 'exclude']:
            parser.error('{} is not a valid option for filter_criteria'
                         .format(_criteria))

        if not _distance.is_integer():
            parser.error("Distance must be an int. {} is not a valid option.")
        if _distance < 0:
            parser.error("Distance should be positive. {} is not a valid "
                         "option.".format(_distance))

    roi_opt_list = []

    # 1) Three-values criteria (or 4 with distance)
    # a) Get them all
    for roi_opt in drawn_roi:
        roi_opt_list.append(['drawn_roi'] + roi_opt)
    for roi_opt in bdo:
        roi_opt_list.append(['bdo'] + roi_opt)
    for roi_opt in x_plane:
        if not (0 <= roi_opt[0] < dim[0]):
            parser.error('x_plane is not valid according to the tractogram '
                         'header. Expecting a value between {} and {}'
                         .format(0, dim[0]))
        if not roi_opt.is_integer():
            parser.error("x_plane must be an integer value")
        roi_opt_list.append(['x_plane'] + roi_opt)
    for roi_opt in y_plane:
        if not (0 <= roi_opt[0] < dim[1]):
            parser.error('y_plane is not valid according to the tractogram '
                         'header. Expecting a value between {} and {}'
                         .format(0, dim[1]))
        if not roi_opt.is_integer():
            parser.error("y_plane must be an integer value")
        roi_opt_list.append(['y_plane'] + roi_opt)
    for roi_opt in z_plane:
        if not (0 <= roi_opt[0] < dim[2]):
            parser.error('z_plane is not valid according to the tractogram '
                         'header. Expecting a value between {} and {}'
                         .format(0, dim[2]))
        if not roi_opt.is_integer():
            parser.error("z_plane must be an integer value")
        roi_opt_list.append(['z_plane'] + roi_opt)

    # b) Format to 4-item list, with 0 distance if not specified.
    for roi_opt in roi_opt_list:
        if len(roi_opt) == 4:
            filter_type, file, mode, criteria = roi_opt
            distance = 0
        elif len(roi_opt) == 5:
            filter_type, file, mode, criteria, distance = roi_opt
        else:
            raise ValueError("Please specify 3 or 4 values for {}."
                             .format(roi_opt[1]))
        _check_values(mode, criteria, distance)

    # 2) Four-values criteria (atlas_roi)
    # Get them all and format to 5-item list, with 0 distance if not specified.
    for roi_opt in atlas_roi:
        if len(roi_opt) == 4:
            atlas_id, file, mode, criteria = roi_opt
            distance = 0
        elif len(roi_opt) == 5:
            atlas_id, file, mode, criteria, distance = roi_opt
        else:
            raise ValueError("Please specify 4 or 5 values for atlas_roi.")
        _check_values(mode, criteria, distance)
        roi_opt_list.append(['atlas_roi'] + roi_opt)

    return roi_opt_list


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    args = _convert_filering_list_to_roi_args(parser, args)

    # Get all filenames
    roi_files_with_header = [drawn_roi[0] for drawn_roi in args.drawn_roi] + \
                            [atlas_roi[0] for atlas_roi in args.atlas_roi]
    roi_files_no_header = [bdo[0] for bdo in args.bdo]

    # Any existing atlas roi:
    other_outputs = []
    if args.extract_masks_atlas_roi:
        other_outputs = glob.glob(args.extract_masks_atlas_roi + '*.nii.gz')

    assert_inputs_exist(parser, args.in_tractogram,
                        roi_files_with_header + roi_files_no_header +
                        [args.reference, args.filtering_list])
    assert_outputs_exist(parser, args, args.out_tractogram,
                         [args.save_rejected] + other_outputs)
    assert_headers_compatible(parser, args.in_tractogram,
                              roi_files_with_header + [args.filtering_list],
                              reference=args.reference)

    dict_overwrite_distance = _read_and_check_overwrite_distance(parser, args)

    # Loading
    # Other files (ROIs) will be loaded on-the-fly
    logging.info("Loading the tractogram...")
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    # Can now verify that all values given by user are ok.
    _, dim, _, _ = sft.space_attributes
    roi_opt_list = _prepare_filtering_criteria(
        parser, args.drawn_roi, args.atlas_roi, args.bdo,
        args.x_plane, args.y_plane, args.z_plane, dim)

    if args.save_rejected:
        initial_sft = deepcopy(sft)

    # Processing

    o_dict = {'streamline_count_before_filtering': len(sft.streamlines)}

    atlas_roi_item = 0
    total_kept_ids = np.arange(len(sft.streamlines))
    for i, roi_opt in enumerate(roi_opt_list):
        logging.info("Preparing filtering from option: {}".format(roi_opt))

        plane_id = None
        atlas_id = None
        roi_file = None
        if roi_opt[0] == 'atlas_roi':
            filter_type, roi_file, atlas_id, mode, criteria, distance = roi_opt
            roi_file = os.path.abspath(roi_file)
        elif roi_opt[0] in ['drawn_roi', 'bdo']:
            filter_type, roi_file, mode, criteria, distance = roi_opt
            roi_file = os.path.abspath(roi_file)
        else:
            filter_type, plane_id, mode, criteria, distance = roi_opt

        # Overwrite distance?
        key_distance = '-'.join([mode, criteria])
        if key_distance in dict_overwrite_distance:
            distance = dict_overwrite_distance[key_distance]
        else:
            distance = distance

        is_exclude = True if criteria == 'exclude' else False

        if filter_type == 'drawn_roi' or filter_type == 'atlas_roi':
            # FILTERING FROM ROI
            img = nib.load(roi_file)
            if filter_type == 'drawn_roi':
                mask = get_data_as_mask(img)
            else:
                atlas = get_data_as_labels(img)
                mask = merge_labels_into_mask(atlas, atlas_id)

                if args.extract_masks_atlas_roi:
                    atlas_roi_item += 1  # Counting how many files.
                    filename = args.extract_masks_atlas_roi + \
                        '{}.nii.gz'.format(atlas_roi_item)
                    img = nib.Nifti1Image(mask, img.affine)
                    img.to_filename(filename)

            filtered_sft, kept_ids = filter_grid_roi(
                sft, mask, mode, is_exclude, distance)

        elif filter_type in ['x_plane', 'y_plane', 'z_plane']:
            # FILTERING FROM PLANE
            plane_id = int(plane_id)
            mask = np.zeros(dim, dtype=np.int16)
            if filter_type == 'x_plane':
                mask[plane_id, :, :] = 1
            elif filter_type == 'y_plane':
                mask[:, plane_id, :] = 1
            elif filter_type == 'z_plane':
                mask[:, :, plane_id] = 1

            filtered_sft, kept_ids = filter_grid_roi(
                sft, mask, mode, is_exclude, distance)

        else:  # filter_type == 'bdo':
            # FILTERING FROM BOUNDING BOX
            geometry, radius, center = read_info_from_mb_bdo(roi_file)

            if distance != 0:
                radius += distance * sft.space_attributes[2]

            if geometry == 'Ellipsoid':
                filtered_sft, kept_ids = filter_ellipsoid(
                    sft, radius, center, mode, is_exclude)
            else:  # geometry == 'Cuboid':
                filtered_sft, kept_ids = filter_cuboid(
                    sft, radius, center, mode, is_exclude)

        logging.info('The filtering options {} resulted in {} included '
                     'streamlines'.format(roi_opt, len(filtered_sft)))

        sft = filtered_sft
        total_kept_ids = total_kept_ids[kept_ids]
        o_dict['streamline_count_after_criteria{}'.format(i)] = \
            len(sft.streamlines)

    # Streamline count after filtering
    o_dict['streamline_count_final_filtering'] = len(sft.streamlines)
    if args.display_counts:
        print(json.dumps(o_dict, indent=args.indent))

    save_tractogram(sft, args.out_tractogram, args.no_empty)

    if args.save_rejected:
        rejected_ids = np.setdiff1d(np.arange(len(initial_sft.streamlines)),
                                    total_kept_ids)

        sft = initial_sft[rejected_ids]
        save_tractogram(sft, args.save_rejected, args.no_empty)


if __name__ == "__main__":
    main()
