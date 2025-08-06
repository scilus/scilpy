#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
This script filters streamlines in a tractogram according to their geometrical
properties (i.e. limiting their length and looping angle) and their anatomical
ending properties (i.e. the anatomical tissue or region their endpoints lie
in).

See also:
    - scil_tractogram_detect_loops.py
    - scil_tractogram_filter_by_length.py
    - scil_tractogram_filter_by_orientation.py
    - scil_tractogram_filter_by_roi.py
    - scil_labels_dilate.py
    (This script is a combination of all these scripts, in one step).

The filtering is performed sequentially in four steps, each step processing the
data on the output of the previous step:

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
             ``--angle`` option.

Length and loop-based filtering (steps 1 and 2) will not have practical effects
if no specific thresholds are provided (but will be still executed), since
default values are 0 for the minimum allowed length and infinite for the
maximum allowed length and angle.

The anatomical region endings filtering requires a parcellation or label image
file including the cerebrospinal fluid and gray matter (cortical) regions
according to the Desikan-Killiany atlas. Intermediate tractograms (results of
each step and outliers) and volumes can be saved throughout the process.

Usage examples:

-  Filter out "bad streamlines" using length, looping angle and anatomical
ending region (must end in the gray matter).
>>> scil_tractogram_filter_by_anatomy.py tractogram.trk wmparc.nii.gz
    path/to/output/directory --minL 20 --maxL 200 --angle 300

- Filter out "bad streamline" using only anatomical ending region, with WM
dilation and provided CSF mask.
>>> scil_tractogram_filter_by_anatomy.py tractogram.trk wmparc.nii.gz
    path/to/output/directory --csf_bin csf_bin.nii.gz --ctx_dilation_radius 2

Formerly: scil_filter_streamlines_anatomically.py

NOTE: As of version 2.0.0, the dilation of the cortical mask has changed; from
an in-house process to scipy's dilation. Results may differ from previous
versions.
----------------------------------------------------------------------------
References:
[1] Jörgens, D., Descoteaux, M., Moreno, R., 2021. Challenges for
    tractogram ﬁltering. In: Özarslan, E., Schultz, T., Zhang, E., Fuster,
    A. (Eds.), Anisotropy Across Fields and Scales. Springer. Mathematics
    and Visualization.

[2] Legarreta, J., Petit, L., Rheault, F., Theaud, G., Lemaire, C.,
    Descoteaux, M., Jodoin, P.M. Filtering in tractography using
    autoencoders (FINTA). Medical Image Analysis. 2021
----------------------------------------------------------------------------
"""

import argparse
from copy import deepcopy
import json
import logging
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

from scilpy.io.streamlines import (load_tractogram_with_reference,
                                   save_tractogram)
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_json_args, add_overwrite_arg,
                             add_processes_arg, add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             validate_nbr_processes, assert_headers_compatible,
                             ranged_type)
from scilpy.image.labels import (get_data_as_labels, load_wmparc_labels,
                                 get_binary_mask_from_labels)
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.tractograms.streamline_operations import \
    filter_streamlines_by_length, remove_loops_and_sharp_turns
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file.')
    p.add_argument('in_wmparc',
                   help='Path of the white matter parcellation atlas '
                        '(.nii or .nii.gz)')
    p.add_argument('out_path',
                   help='Path to the output files.')

    p.add_argument('--minL', default=0., type=ranged_type(float, 0, None),
                   help='Minimum length of streamlines, in mm. [%(default)s]')
    p.add_argument('--maxL', default=np.inf, type=ranged_type(float, 0, None),
                   help='Maximum length of streamlines, in mm. [%(default)s]')
    p.add_argument('--angle', default=np.inf, type=ranged_type(float, 0, None),
                   help='Maximum looping (or turning) angle of a streamline, '
                        '\nin degrees. [%(default)s]')

    p.add_argument('--csf_bin',
                   help='Allow CSF endings filtering with this binary\n'
                        'mask instead of using the atlas (.nii or .nii.gz)')
    p.add_argument('--dilate_ctx', metavar='value',
                   type=ranged_type(int, 1, None),
                   help='If set, dilate the cortical labels. Value is the '
                        'dilation \nradius, in voxels (an integer > 0)')
    p.add_argument('--save_intermediate_tractograms', action='store_true',
                   help='Save accepted and discarded streamlines after each '
                        'step.')
    p.add_argument('--save_volumes', action='store_true',
                   help='Save volumetric images (e.g. binarised label \n'
                        'images, etc) in the filtering process.')
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


def _finalize_step(args, sft, outliers_sft, step_nb, step_name,
                   steps_combined, in_sft_name, ext, o_dict):
    """
    Save the clean stateful tractogram in:
        out_path/nb-{step_name}//{in_sft_name}_{steps_combine}.{ext}

    Save the outliers (of this step only) in:
        out_path/nb-{step_name}//{in_sft_name}_{steps_combine}_outliers.{ext}
    """
    intermediate_name = in_sft_name + '_' + steps_combined + ext
    intermediate_outliers_name = \
        in_sft_name + '_' + steps_combined + '_outliers' + ext

    # Save count
    o_dict[intermediate_name] = {'streamline_count': len(sft.streamlines)}
    o_dict[intermediate_outliers_name] = \
        {'streamline_count': len(outliers_sft.streamlines)}

    # Save intermediate clean tractogram and outliers.
    if args.save_intermediate_tractograms:
        new_path = _create_subdir(args.out_path, step_nb + step_name)

        sft_filename = os.path.join(new_path, intermediate_name)
        outliers_filename = os.path.join(new_path, intermediate_outliers_name)

        save_tractogram(sft, sft_filename, args.no_empty)
        save_tractogram(outliers_sft, outliers_filename, args.no_empty)


def _finish_all(args, final_sft, total_outliers, o_dict, out_sft_name,
                rejected_sft_name):
    """
    Finish now if no streamlines left.
    """
    if args.verbose == "INFO" or args.verbose == "DEBUG":
        o_dict_str = json.dumps(o_dict, indent=args.indent,
                                sort_keys=args.sort_keys)
        logging.info("Streamline count:\n{}".format(o_dict_str))
    if args.save_counts:
        fname = os.path.join(args.out_path, "streamline_count.json")
        with open(fname, 'w') as outfile:
            json.dump(o_dict, outfile, indent=args.indent,
                      sort_keys=args.sort_keys)

    save_tractogram(final_sft, out_sft_name, args.no_empty)
    if args.save_rejected:
        save_tractogram(total_outliers, rejected_sft_name, args.no_empty)


def _create_subdir(out_path, dir_name):
    """
    Create a directory named ``dir_name`` at ``out_path``
    """
    new_path = os.path.join(out_path, dir_name)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, [args.in_tractogram, args.in_wmparc],
                        [args.csf_bin, args.reference])
    assert_output_dirs_exist_and_empty(parser, args, args.out_path,
                                       create_dir=True)
    assert_headers_compatible(parser, [args.in_tractogram, args.in_wmparc],
                              args.csf_bin, reference=args.reference)

    nbr_cpu = validate_nbr_processes(parser, args)

    if args.minL == 0 and np.isinf(args.maxL):
        logging.info("You have not specified minL nor maxL. Output will "
                     "not be filtered according to length!")
    if np.isinf(args.angle):
        logging.info("You have not specified the angle. Loops will "
                     "not be filtered!")
    if args.dilate_ctx is None:
        logging.info("You have not chosen a value for --dilate_ctx. "
                     "The wmparc atlas will not be dilated!")

    # Prepare output names
    in_sft_name, ext = os.path.splitext(os.path.basename(args.in_tractogram))
    out_sft_rootname = in_sft_name + "_filtered"
    out_sft_name = os.path.join(args.out_path, out_sft_rootname + ext)

    # Loading
    img_wmparc = nib.load(args.in_wmparc)
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    wm_labels = load_wmparc_labels()  # Loads labels from our own data

    # More loadings to come. But waiting to see that they are necessary to
    # load them.

    # Processing!

    step_dict = ['length', 'no_csf', 'end_in_atlas', 'no_loops']

    # Remember initial SFT
    sft.to_vox()
    sft.to_corner()
    rejected_sft_name = None
    if args.save_rejected:
        rejected_sft_name = os.path.join(args.out_path,
                                         in_sft_name + "_rejected" + ext)

    # o_dict will be a dict with the streamline_count at each step.
    # Initial count:
    o_dict = {
        in_sft_name + ext: {'streamline_count': len(sft.streamlines)}
    }

    # STEP 1 - Filter length
    step = step_dict[0]
    steps_combined = step
    step_nb = '01-'
    logging.info("STEP 1: Filtering by length: {} - {} mm"
                 .format(args.minL, args.maxL))
    sft, _, outliers_sft = filter_streamlines_by_length(
        sft, args.minL, args.maxL, return_rejected=True)
    logging.info("  -> Found {} good streamlines (rejected {})"
                 .format(len(sft), len(outliers_sft)))
    _finalize_step(args, sft, outliers_sft, step_nb, step,
                   steps_combined, in_sft_name, ext, o_dict)
    total_outliers = outliers_sft
    if len(sft.streamlines) == 0:
        _finish_all(args, sft, total_outliers, o_dict, out_sft_name,
                    rejected_sft_name)
        exit(0)

    # STEP 2 - Filter CSF
    step = step_dict[1]
    steps_combined += "_" + step
    step_nb = '02-'
    logging.info("STEP 2: Filtering out streamlines ending in the CSF.")

    # Loading mask now
    if args.csf_bin:
        csf_mask = get_data_as_mask(nib.load(args.csf_bin))
    else:
        atlas = get_data_as_labels(img_wmparc)
        csf_mask = get_binary_mask_from_labels(atlas, wm_labels["csf_labels"])

        if args.save_volumes:
            new_path = _create_subdir(args.out_path, step_nb + step)
            nib.save(nib.Nifti1Image(csf_mask, img_wmparc.affine,
                                     img_wmparc.header),
                     os.path.join(new_path, 'csf_bin.nii.gz'))

    _, sft, outliers_sft = filter_grid_roi(sft, csf_mask, 'any',
                                           is_exclude=True,
                                           return_sft=True,
                                           return_rejected_sft=True)
    logging.info("  -> Found {} good streamlines (rejected {})"
                 .format(len(sft), len(outliers_sft)))
    _finalize_step(args, sft, outliers_sft, step_nb, step,
                   steps_combined, in_sft_name, ext, o_dict)
    total_outliers += outliers_sft
    if len(sft.streamlines) == 0:
        _finish_all(args, sft, total_outliers, o_dict, out_sft_name,
                    rejected_sft_name)
        exit(0)

    # STEP 3 - Filter WM endings
    step = step_dict[2]
    steps_combined += "_" + step
    step_nb = '03-'
    logging.info("STEP 3: Filtering out streamlines ending in the WM.")
    logging.debug("(i.e. not in the GM, based on ctx_lh_fs_labels: {}, "
                  "ctx_rh_fs_labels: {} or nuclei_fs_labels: {})"
                  .format(wm_labels["ctx_lh_fs_labels"],
                          wm_labels["ctx_rh_fs_labels"],
                          wm_labels["nuclei_fs_labels"]))

    # Mask creation
    ctx_fs_labels = (wm_labels["ctx_lh_fs_labels"] +
                     wm_labels["ctx_rh_fs_labels"])
    atlas_wm = get_data_as_labels(img_wmparc)
    wmparc_ctx = get_binary_mask_from_labels(atlas_wm, ctx_fs_labels)
    wmparc_nuclei = get_binary_mask_from_labels(atlas_wm,
                                                wm_labels["nuclei_fs_labels"])

    # Dilation of cortex
    if args.dilate_ctx is not None:
        ctx_mask = binary_dilation(wmparc_ctx, iterations=args.dilate_ctx)
    else:
        ctx_mask = wmparc_ctx

    gm_mask = np.zeros(atlas_wm.shape, dtype=np.uint16)
    gm_mask[np.logical_or(wmparc_nuclei, ctx_mask)] = 1

    if args.save_volumes:
        new_path = _create_subdir(args.out_path, step_nb + step)
        nib.save(nib.Nifti1Image(gm_mask, img_wmparc.affine,
                                 img_wmparc.header),
                 os.path.join(new_path, 'atlas_bin.nii.gz'))

    # Filter tractogram
    _, sft, outliers_sft = filter_grid_roi(sft, gm_mask, 'both_ends',
                                           is_exclude=False,
                                           return_sft=True,
                                           return_rejected_sft=True)
    logging.info("  -> Found {} good streamlines (rejected {})"
                 .format(len(sft), len(outliers_sft)))
    _finalize_step(args, sft, outliers_sft, step_nb, step,
                   steps_combined, in_sft_name, ext, o_dict)
    total_outliers += outliers_sft
    if len(sft.streamlines) == 0:
        _finish_all(args, sft, total_outliers, o_dict, out_sft_name,
                    rejected_sft_name)
        exit(0)

    # STEP 4 - Filter loops
    step = step_dict[3]
    steps_combined += "_" + step
    step_nb = '04-'

    logging.info("STEP 4: Filtering loops and sharp turns.")
    if args.angle != np.inf:
        ids_c = remove_loops_and_sharp_turns(sft.streamlines, args.angle,
                                             num_processes=nbr_cpu)
        sft = sft[ids_c]
    else:
        ids_c = np.arange(len(sft))
        sft = deepcopy(sft)

    outliers_ids = np.setdiff1d(np.arange(len(sft)), ids_c)
    outliers_sft = sft[outliers_ids]
    logging.info("  -> Found {} good streamlines (rejected {})"
                 .format(len(sft), len(outliers_sft)))
    _finalize_step(args, sft, outliers_sft, step_nb, step, steps_combined,
                   in_sft_name, ext, o_dict)

    # Final saving
    total_outliers += outliers_sft
    _finish_all(args, sft, total_outliers, o_dict, out_sft_name,
                rejected_sft_name)


if __name__ == "__main__":
    main()
