#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the angular error (AE) for each segment of the streamlines.

For each segment of each streamline, the direction is compared with the
underlying peak (for single peak files, e.g. the first eigen-vector of tensors
for DTI) or with the closest peak (ex, with fODF peaks). AE is computed as the
cosine difference.

Currently, interpolation is not supported: peaks of  the closest voxel are used
(nearest neighbor).

The AE is added as data_per_point (dpp) for each point, using the first point
of the segment. The last point of each streamline has an AE of zero.
Optionnally, you may also save it as a color.

When using --save_mean_map, if you want to be sure that your streamlines have
points in most voxels that they touch, you could resample your tractogram
first, with a small step size. See scil_tractogram_resample_nb_points.

"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import (load_tractogram_with_reference, 
                                   save_tractogram)
from scilpy.io.utils import (add_processes_arg, add_verbose_arg, 
                             add_overwrite_arg, assert_headers_compatible, 
                             assert_inputs_exist, assert_outputs_exist, 
                             add_bbox_arg)
from scilpy.tractanalysis.scoring import compute_ae
from scilpy.tractograms.dps_and_dpp_management import (add_data_as_color_dpp, 
                                                       project_dpp_to_map)
from scilpy.version import version_string
from scilpy.viz.color import get_lookup_table


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file (trk or tck).')
    p.add_argument('in_peaks',
                   help='Path of the input peaks file.')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file (trk or tck).')
    p.add_argument('--dpp_key', default="AE",
                   help="Name of the dpp key containg the AE in the output. "
                        "Default: AE")
    
    g = p.add_argument_group("Optional outputs")
    g.add_argument('--save_as_color', action='store_true',
                   help="Save the AE as a color. Colors will range between "
                        "black (0) and yellow (--cmax_max) \n"
                        "See also scil_tractogram_assign_custom_color, option "
                        "--use_dpp.")
    g.add_argument('--save_mean_map', metavar='filename',
                   help="If set, save the mean value of each streamline per "
                        "voxel. Name of the map file (nifti).\n"
                        "See also scil_tractogram_project_streamlines_to_map.")
    g.add_argument('--save_worst', metavar='filename',
                   help="If set, save the worst streamlines in a separate "
                        "tractogram.")

    g = p.add_argument_group("Processing options")
    g.add_argument('--cmap_max', nargs='?', const=180,
                    help="If set, the maximum color on the colormap (yellow) "
                         "will be associated \nto this value. If not set, the "
                         "maxium value found in the data will be used instead. "
                         "Default if set: 180 degrees.")

    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_bbox_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # -- Verifications
    args.reference = args.in_peaks
    assert_inputs_exist(parser, [args.in_tractogram, args.in_peaks],
                        args.reference)
    assert_headers_compatible(parser, [args.in_tractogram, args.in_peaks], [],
                              args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram, 
                        [args.save_mean_map, args.save_worst])

    # -- Loading
    peaks = nib.load(args.in_peaks).get_fdata()
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    logging.info("Loaded data")

    # Removing invalid
    len_before = len(sft)
    sft.remove_invalid_streamlines()
    len_after = len(sft)
    if len_before != len_after:
        logging.warning("Removed {} invalid streamlines before processing"
                        .format(len_before - len_after))

    # Verify if the key already exists
    if args.dpp_key in sft.data_per_point.keys() and not args.overwrite:
        parser.error("--dpp_key already exists. Use --overwrite to proceed.")
    if (args.save_as_color and 'color' in sft.data_per_point.keys() and
            not args.overwrite):
        parser.error("The 'color' dpp already exists. Use --overwrite to "
                     "proceed.")

    # -- Processing
    ae = compute_ae(sft, peaks, nb_processes=args.nbr_processes)

    # Printing stats
    stacked_ae = np.hstack(ae)
    mean_ae = np.mean(stacked_ae)
    std_ae = np.std(stacked_ae)
    min_ae = np.min(stacked_ae)
    max_ae = np.max(stacked_ae)
    logging.info("AE computed. Some statistics:\n"
                 "- Mean AE: {} +- {} \n"
                 "- Range:[{}, {}]".format(mean_ae, std_ae, min_ae, max_ae))

    # Add as dpp
    ae_dpp = [ae_s[:, None] for ae_s in ae]
    sft.data_per_point[args.dpp_key] = ae_dpp

    # Add as color (optional)
    if args.save_as_color:
        max_cmap = args.cmap_max if args.cmap_max is not None \
            else np.max(stacked_ae)
        logging.info("Saving colors. The maxium color is assiociated to "
                     "value {}".format(max_cmap))
         
        cmap = get_lookup_table('jet')
        sft, _, _ = add_data_as_color_dpp(sft, cmap, stacked_ae,
                                          min_cmap=0, max_cmap=max_cmap)

    # -- Saving
    logging.info("Saving file {}.".format(args.out_tractogram))
    save_tractogram(sft, args.out_tractogram, no_empty=False)
    
    # Save map (optional)
    if args.save_mean_map is not None:
        logging.info("Preparing map.")
        the_map = project_dpp_to_map(sft, args.dpp_key)

        logging.info("Saving file {}".format(args.save_mean_map))
        nib.save(nib.Nifti1Image(the_map, sft.affine), args.save_mean_map)


if __name__ == "__main__":
    main()
