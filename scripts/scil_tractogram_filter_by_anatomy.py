#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
This script filters streamlines in a tractogram according to their geometrical
properties (i.e. limiting their length and looping angle) and their anatomical
ending properties (i.e. the anatomical tissue or region their endpoints lie
in). The filtering is performed sequentially in four steps, each step
processing the data on the output of the previous step:

    Step 1 - Remove streamlines below the minimum length and above the
             maximum length. These thresholds must be set with the ``--minL``
             and ``--maxL`` options.
    Step 2 - Ensure that no streamlines end in the cerebrospinal fluid
             according to the provided parcellation. A binary mask can be used
             alternatively through the ``--csf_bin`` option.
    Step 3 - Ensure that no streamlines end in white matter by ensuring that
             they reach the cortical regions according to the provided
             parcellation. The cortical regions of the parcellation can be
             dilated using the ``--ctx_dilation_radius``.
    Step 4 - Remove streamlines if they make a loop with an angle above a
             certain threshold. It's possible to change this angle with the
             ``-a`` option.

Length and loop-based filtering (steps 1 and 2) will not have practical effects
if no specific thresholds are provided (but will be still executed), since
default values are 0 for the minimum allowed length and infinite for the
maximum allowed length and angle.

The anatomical region endings filtering requires a parcellation or label image
file including the cerebrospinal fluid and gray matter (cortical) regions
according to the Desikan-Killiany atlas. Intermediate tractograms (results of
each step and outliers) and volumes can be saved throughout the process.

Example usages:

# Filter length, looping angle and anatomical ending region
>>> scil_tractogram_filter_by_anatomy.py tractogram.trk wmparc.nii.gz
    path/to/output/directory --minL 20 --maxL 200 -a 300
# Filter only anatomical ending region, with WM dilation and provided csf mask
>>> scil_tractogram_filter_by_anatomy.py tractogram.trk wmparc.nii.gz
    path/to/output/directory --csf_bin csf_bin.nii.gz --ctx_dilation_radius 2

Formerly: scil_filter_streamlines_anatomically.py
"""

import argparse
from copy import deepcopy
import json
import logging
import os
import importlib.resources as resources

from dipy.io.streamline import save_tractogram
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_processes_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             validate_nbr_processes, assert_headers_compatible)
from scilpy.image.labels import get_data_as_labels
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.tractograms.streamline_operations import \
    filter_streamlines_by_length, remove_loops_and_sharp_turns
from scilpy.tractograms.tractogram_operations import \
    perform_tractogram_operation_on_sft


EPILOG = """
    References:
        [1] Jörgens, D., Descoteaux, M., Moreno, R., 2021. Challenges for
        tractogram ﬁltering. In: Özarslan, E., Schultz, T., Zhang, E., Fuster,
        A. (Eds.), Anisotropy Across Fields and Scales. Springer. Mathematics
        and Visualization.
        [2] Legarreta, J., Petit, L., Rheault, F., Theaud, G., Lemaire, C.,
        Descoteaux, M., Jodoin, P.M. Filtering in tractography using
        autoencoders (FINTA). Medical Image Analysis. 2021
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                epilog=EPILOG, description=__doc__)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file.')
    p.add_argument('in_wmparc',
                   help='Path of the white matter parcellation atlas\n' +
                        '(.nii or .nii.gz)')
    p.add_argument('out_path',
                   help='Path to the output files.')

    p.add_argument('--minL', default=0., type=float,
                   help='Minimum length of streamlines, in mm. [%(default)s]')
    p.add_argument('--maxL', default=np.inf, type=float,
                   help='Maximum length of streamlines, in mm. [%(default)s]')
    p.add_argument('-a', dest='angle', default=np.inf, type=float,
                   help='Maximum looping (or turning) angle of\n' +
                        'a streamline, in degrees. [%(default)s]')

    p.add_argument('--csf_bin',
                   help='Allow CSF endings filtering with this binary\n' +
                        'mask instead of using the atlas (.nii or .nii.gz)')
    p.add_argument('--ctx_dilation_radius', type=float, default=0.,
                   help='Cortical labels dilation radius, in mm.\n' +
                        ' [%(default)s]')
    p.add_argument('--save_intermediate_tractograms', action='store_true',
                   help='Save accepted and discarded streamlines\n' +
                        ' after each step.')
    p.add_argument('--save_volumes', action='store_true',
                   help='Save volumetric images (e.g. binarised label\n' +
                        ' images, etc) in the filtering process.')
    p.add_argument('--save_counts', action='store_true',
                   help='Save the streamline counts to a file (.json)')
    p.add_argument('--save_rejected', action='store_true',
                   help='Save rejected streamlines to output tractogram.')
    p.add_argument('--no_empty', action='store_true',
                   help='Do not write file if there is no streamlines.')

    add_json_args(p)
    add_processes_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def load_wmparc_labels():
    """
    Load labels dictionary of different parcellations from the
    Desikan-Killiany atlas
    """
    lut_package = resources.files('data').joinpath('LUT')
    labels_path = lut_package.joinpath('dk_aggregate_structures.json')
    with open(labels_path) as labels_file:
        labels = json.load(labels_file)
    return labels


def binarize_labels(atlas, label_list):
    """
    Create a binary mask from specific labels in an atlas (numpy array)
    """
    mask = np.zeros(atlas.shape, dtype=np.uint16)
    for label in label_list:
        is_label = atlas == label
        mask[is_label] = 1

    return mask


def dilate_mask(mask, mask_shape, vox_size, radius):
    """
    Dilate the foreground in a binary mask according to a radius (in mm)
    """
    is_to_dilate = mask == 1
    is_background = mask == 0

    # Get the list of indices
    background_pos = np.argwhere(is_background) * vox_size
    label_pos = np.argwhere(is_to_dilate) * vox_size
    ckd_tree = cKDTree(label_pos)

    # Compute the nearest labels for each voxel of the background
    dist, indices = ckd_tree.query(
        background_pos, k=1, distance_upper_bound=radius,
        workers=-1)

    # Associate indices to the nearest label (in distance)
    valid_nearest = np.squeeze(np.isfinite(dist))
    id_background = np.flatnonzero(is_background)[valid_nearest]
    id_label = np.flatnonzero(is_to_dilate)[indices[valid_nearest]]

    # Change values of those background
    mask = mask.flatten()
    mask[id_background.T] = mask[id_label.T]
    mask = mask.reshape(mask_shape)

    return mask


def create_dir(out_path, dir_name):
    """
    Create a directory named ``dir_name`` at ``out_path``
    """
    new_path = os.path.join(out_path, dir_name)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path


def save_intermediate_sft(sft, outliers_sft, new_path, in_sft_name,
                          step_name, steps_combined, ext, no_empty):
    """
    Save the provided stateful tractograms.
    """
    sft_name = os.path.join(new_path, in_sft_name + "_" + steps_combined + ext)
    outliers_sft_name = os.path.join(
        new_path, in_sft_name + "_" + steps_combined + "_outliers" + ext)

    if len(sft.streamlines) == 0:
        if no_empty:
            logging.info("The file" + sft_name +
                         " won't be written (0 streamlines)")
        save_tractogram(sft, sft_name)
    else:
        save_tractogram(sft, sft_name)

    if len(outliers_sft.streamlines):
        if no_empty:
            logging.info("The file" + outliers_sft_name +
                         " won't be written (0 streamlines)")
        save_tractogram(outliers_sft, outliers_sft_name)
    else:
        save_tractogram(outliers_sft, outliers_sft_name)


def compute_outliers(sft, new_sft):
    """
    Return a stateful tractogram whose streamlines are the difference of the
    two input stateful tractograms
    """
    outliers_sft, _ = perform_tractogram_operation_on_sft('difference_robust',
                                                          [sft, new_sft],
                                                          precision=3,
                                                          no_metadata=True,
                                                          fake_metadata=False)
    return outliers_sft


def save_rejected(sft, new_sft, rejected_sft_name, no_empty):
    """
    Save rejected streamlines
    """
    rejected_sft = compute_outliers(sft, new_sft)

    if len(rejected_sft.streamlines) == 0:
        if no_empty:
            logging.info("The file" + rejected_sft_name +
                         " won't be written (0 streamlines)")
            return

    save_tractogram(rejected_sft, rejected_sft_name)


def display_count(o_dict, indent, sort_keys):
    """
    Display the streamline count.
    """
    o_dict_str = json.dumps(o_dict, indent=indent, sort_keys=sort_keys)
    logging.info("Streamline count:\n{}".format(o_dict_str))


def save_count(o_dict, out_path, indent, sort_keys):
    """
    Save the streamline count to a JSON file.
    """
    fname = os.path.join(out_path, "streamline_count.json")
    with open(fname, 'w') as outfile:
        json.dump(o_dict, outfile, indent=indent, sort_keys=sort_keys)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_tractogram, args.in_wmparc],
                        [args.csf_bin, args.reference])
    assert_output_dirs_exist_and_empty(parser, args, args.out_path,
                                       create_dir=True)
    assert_headers_compatible(parser, [args.in_tractogram, args.in_wmparc],
                              args.csf_bin, reference=args.reference)

    nbr_cpu = validate_nbr_processes(parser, args)

    if args.angle <= 0:
        parser.error('Angle "{}" '.format(args.angle) +
                     'must be greater than or equal to 0')
    if args.ctx_dilation_radius < 0:
        parser.error('Cortex dilation radius "{}" '.format(
                     args.ctx_dilation_radius) + 'must be greater than 0')
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_vox()
    sft.to_corner()

    img_wmparc = nib.load(args.in_wmparc)
    if args.csf_bin:
        img_csf = nib.load(args.csf_bin)

    if args.minL == 0 and np.isinf(args.maxL):
        logging.info("You have not specified minL nor maxL. Output will "
                     "not be filtered according to length!")
    if np.isinf(args.angle):
        logging.info("You have not specified the angle. Loops will "
                     "not be filtered!")
    if args.ctx_dilation_radius == 0:
        logging.info("You have not specified the cortex dilation radius. "
                     "The wmparc atlas will not be dilated!")

    o_dict = {}
    step_dict = ['length', 'no_csf', 'end_in_atlas', 'no_loops']
    wm_labels = load_wmparc_labels()

    in_sft_name = os.path.splitext(os.path.basename(args.in_tractogram))[0]
    out_sft_rootname = in_sft_name + "_filtered"
    _, ext = os.path.splitext(args.in_tractogram)
    out_sft_name = os.path.join(args.out_path,
                                out_sft_rootname + ext)

    if args.save_rejected:
        initial_sft = deepcopy(sft)
        rejected_sft_name = os.path.join(args.out_path,
                                         in_sft_name +
                                         "_rejected" + ext)

    # STEP 1 - Filter length
    step = step_dict[0]
    steps_combined = step
    new_sft = filter_streamlines_by_length(sft, args.minL, args.maxL)
    # Streamline count before and after filtering lengths
    o_dict[in_sft_name + ext] =\
        dict({'streamline_count': len(sft.streamlines)})
    o_dict[in_sft_name + '_' + steps_combined + ext] =\
        dict({'streamline_count': len(new_sft.streamlines)})

    if args.save_intermediate_tractograms:
        outliers_sft = compute_outliers(sft, new_sft)
        new_path = create_dir(args.out_path, '01-' + step)
        save_intermediate_sft(new_sft, outliers_sft, new_path, in_sft_name,
                              step, steps_combined, ext, args.no_empty)
        o_dict[in_sft_name + '_' + steps_combined + '_outliers' + ext] =\
            dict({'streamline_count': len(outliers_sft.streamlines)})

    if len(new_sft.streamlines) == 0:
        if args.no_empty:
            logging.info("The file {} won't be written".format(
                         out_sft_name) + "(0 streamlines after "
                         + step + " filtering).")

            if args.verbose:
                display_count(o_dict, args.indent, args.sort_keys)
            if args.save_counts:
                save_count(o_dict, args.out_path, args.indent, args.sort_keys)
            if args.save_rejected:
                save_tractogram(initial_sft, rejected_sft_name)
            return

        logging.info('The file {} contains 0 streamlines after '.format(
                     out_sft_name) + step + ' filtering')
        save_tractogram(new_sft, out_sft_name)

        if args.save_rejected:
            save_rejected(initial_sft, new_sft,
                          rejected_sft_name, args.no_empty)
        if args.verbose:
            display_count(o_dict, args.indent, args.sort_keys)
        if args.save_counts:
            save_count(o_dict, args.out_path, args.indent, args.sort_keys)
        return

    sft = new_sft

    # STEP 2 - Filter CSF
    step = step_dict[1]
    steps_combined += "_" + step

    # Mask creation
    if args.csf_bin:
        mask = get_data_as_mask(img_csf)
    else:
        atlas = get_data_as_labels(img_wmparc)
        mask = binarize_labels(atlas, wm_labels["csf_labels"])

    # Filter tractogram
    new_sft, _ = filter_grid_roi(sft, mask, 'any', True)
    # Streamline count after filtering CSF endings
    o_dict[in_sft_name + '_' + steps_combined + ext] =\
        dict({'streamline_count': len(new_sft.streamlines)})

    if args.save_volumes:
        new_path = create_dir(args.out_path, '02-' + step)
        if not args.csf_bin:
            nib.save(nib.Nifti1Image(mask, img_wmparc.affine,
                                     img_wmparc.header),
                     os.path.join(new_path, 'csf_bin' + '.nii.gz'))

    if args.save_intermediate_tractograms:
        outliers_sft = compute_outliers(sft, new_sft)
        new_path = create_dir(args.out_path, '02-' + step)
        save_intermediate_sft(new_sft, outliers_sft, new_path, in_sft_name,
                              step, steps_combined, ext, args.no_empty)
        o_dict[in_sft_name + '_' + steps_combined + '_outliers' + ext] =\
            dict({'streamline_count': len(outliers_sft.streamlines)})

    if len(new_sft.streamlines) == 0:
        if args.no_empty:
            logging.info("The file {} won't be written".format(
                         out_sft_name) + "(0 streamlines after "
                         + step + " filtering).")

            if args.verbose:
                display_count(o_dict, args.indent, args.sort_keys)
            if args.save_counts:
                save_count(o_dict, args.out_path, args.indent, args.sort_keys)
            if args.save_rejected:
                save_tractogram(sft, rejected_sft_name)
            return

        logging.info('The file {} contains 0 streamlines after '.format(
                     out_sft_name) + step + ' filtering')
        save_tractogram(new_sft, out_sft_name)

        if args.save_rejected:
            save_rejected(initial_sft, new_sft,
                          rejected_sft_name, args.no_empty)
        if args.verbose:
            display_count(o_dict, args.indent, args.sort_keys)
        if args.save_counts:
            save_count(o_dict, args.out_path, args.indent, args.sort_keys)
        return

    sft = new_sft

    # STEP 3 - Filter WM endings
    step = step_dict[2]
    steps_combined += "_" + step

    # Mask creation
    ctx_fs_labels = wm_labels["ctx_lh_fs_labels"] + \
        wm_labels["ctx_rh_fs_labels"]
    vox_size = np.reshape(img_wmparc.header.get_zooms(), (1, 3))
    atlas_wm = get_data_as_labels(img_wmparc)
    atlas_shape = atlas_wm.shape
    wmparc_ctx = binarize_labels(atlas_wm, ctx_fs_labels)
    wmparc_nuclei = binarize_labels(atlas_wm, wm_labels["nuclei_fs_labels"])

    # Dilation of cortex
    if args.ctx_dilation_radius:
        ctx_mask = dilate_mask(wmparc_ctx, atlas_shape, vox_size,
                               args.ctx_dilation_radius)
    else:
        ctx_mask = wmparc_ctx

    freesurfer_mask = np.zeros(atlas_shape, dtype=np.uint16)
    freesurfer_mask[np.logical_or(wmparc_nuclei, ctx_mask)] = 1

    # Filter tractogram
    new_sft, _ = filter_grid_roi(sft, freesurfer_mask, 'both_ends', False)

    # Streamline count after final filtering
    o_dict[in_sft_name + '_' + steps_combined + ext] =\
        dict({'streamline_count': len(new_sft.streamlines)})

    if args.save_volumes:
        new_path = create_dir(args.out_path, '03-' + step)
        nib.save(nib.Nifti1Image(freesurfer_mask, img_wmparc.affine,
                                 img_wmparc.header),
                 os.path.join(new_path, 'atlas_bin' + '.nii.gz'))

    if args.save_intermediate_tractograms:
        outliers_sft = compute_outliers(sft, new_sft)
        new_path = create_dir(args.out_path, '03-' + step)
        save_intermediate_sft(new_sft, outliers_sft, new_path, in_sft_name,
                              step, steps_combined, ext, args.no_empty)
        o_dict[in_sft_name + '_' + steps_combined + '_outliers' + ext] =\
            dict({'streamline_count': len(outliers_sft.streamlines)})

    if len(new_sft.streamlines) == 0:
        if args.no_empty:
            logging.info("The file {} won't be written".format(
                         out_sft_name) + "(0 streamlines after "
                         + step + " filtering).")

            if args.verbose:
                display_count(o_dict, args.indent, args.sort_keys)
            if args.save_counts:
                save_count(o_dict, args.out_path, args.indent, args.sort_keys)
            if args.save_rejected:
                save_tractogram(sft, rejected_sft_name)
            return

        logging.info('The file {} contains 0 streamlines after '.format(
                     out_sft_name) + step + ' filtering')
        save_tractogram(new_sft, out_sft_name)

        if args.save_rejected:
            save_rejected(initial_sft, new_sft,
                          rejected_sft_name, args.no_empty)
        if args.verbose:
            display_count(o_dict, args.indent, args.sort_keys)
        if args.save_counts:
            save_count(o_dict, args.out_path, args.indent, args.sort_keys)
        return

    sft = new_sft

    # STEP 4 - Filter loops
    step = step_dict[3]
    steps_combined += "_" + step

    if args.angle != np.inf:
        ids_c = remove_loops_and_sharp_turns(sft.streamlines, args.angle,
                                             num_processes=nbr_cpu)
        new_sft = sft[ids_c]
    else:
        new_sft = deepcopy(sft)

    # Streamline count after filtering loops
    o_dict[in_sft_name + '_' + steps_combined + ext] =\
        dict({'streamline_count': len(new_sft.streamlines)})

    if args.save_intermediate_tractograms:
        outliers_sft = compute_outliers(sft, new_sft)
        new_path = create_dir(args.out_path, '04-' + step)
        save_intermediate_sft(new_sft, outliers_sft, new_path, in_sft_name,
                              step, steps_combined, ext, args.no_empty)
        o_dict[in_sft_name + '_' + steps_combined + '_outliers' + ext] =\
            dict({'streamline_count': len(outliers_sft.streamlines)})

    if len(new_sft.streamlines) == 0:
        if args.no_empty:
            logging.info("The file {} won't be written".format(
                         out_sft_name) + "(0 streamlines after "
                         + step + " filtering).")

            if args.verbose:
                display_count(o_dict, args.indent, args.sort_keys)
            if args.save_counts:
                save_count(o_dict, args.out_path, args.indent, args.sort_keys)
            if args.save_rejected:
                save_tractogram(sft, rejected_sft_name)
            return

        logging.info('The file {} contains 0 streamlines after '.format(
                     out_sft_name) + step + ' filtering')
        save_tractogram(new_sft, out_sft_name)

    if args.verbose == "INFO" or args.verbose == "DEBUG":
        display_count(o_dict, args.indent, args.sort_keys)
    if args.save_counts:
        save_count(o_dict, args.out_path, args.indent, args.sort_keys)

    sft = new_sft
    save_tractogram(sft, out_sft_name)
    if args.save_rejected:
        save_rejected(initial_sft, sft, rejected_sft_name, args.no_empty)


if __name__ == "__main__":
    main()
