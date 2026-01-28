#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract some streamlines from chosen criterion based on streamlines' dpp 
(data_per_point) or dps (data_per_streamline).

See also:
    - To modify your dpp / dps values: see scil_tractogram_dpp_math and 
    scil_tractogram_dps_math.
    - To extract streamlines based on regions of interest (ROI), see 
    scil_tractogram_segment_with_ROI.
    - To extract U-shaped streamlines, see scil_tractogram_extract_ushape
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import (load_tractogram_with_reference,
                                   save_tractogram)
from scilpy.io.utils import (add_bbox_arg, add_overwrite_arg, add_reference_arg, 
                             add_verbose_arg, assert_inputs_exist, 
                             assert_outputs_exist, ranged_type)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file (trk or tck).')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file (trk or tck).')
    p.add_argument('--no_empty', action='store_true',
                   help="Do not save the output tractogram if no streamline "
                        "fit the criterion.")
    
    g = p.add_argument_group("Criterion's data")
    gg = g.add_mutually_exclusive_group(required=True)
    gg.add_argument("--from_dps", metavar='dpp_key',
                    help="Use DPS as criteria.")
    gg.add_argument("--from_dpp", metavar='dpp_key',
                    help="Use DPP as criteria. Uses the average value over "
                    "each streamline.")

    g = p.add_argument_group("Direction of the criterion:")
    gg = g.add_mutually_exclusive_group(required=True)
    gg.add_argument('--top', action='store_true',
                    help="If set, selects a portion of streamlines that has "
                         "the highest value in its dps or mean dpp.")
    gg.add_argument('--bottom', action='store_true',
                    help="If set, selects a portion of streamlines that has "
                         "the lowest value in its dps or mean dpp.")
    gg.add_argument('--center', action='store_true',
                    help="Selects the average streamlines.")
    

    g = p.add_argument_group("Criterion")
    gg = g.add_mutually_exclusive_group(required=True)
    gg.add_argument(
        '--nb', type=int,
        help="Selects a chosen number of streamlines.")
    gg.add_argument(
        '--percent', type=ranged_type(float, 0, 100),  const=5,  nargs='?',
        help=r"Saves the streamlines in the top / lowest percentile. "
             "Default if set: The top / bottom 5%")
    gg.add_argument(
        '--mean_std', type=int, const=3,  nargs='?',
        help="Saves the streamlines with value above mean + N*std (option "
             "--top), below\n mean - N*std (option --below) or in the "
             "range [mean - N*std, mean + N*std] )option --center)."
             "Default if set: uses mean +- 3std.")   

    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_bbox_arg(p)
    add_reference_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # -- Verifications
    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram)

    # -- Loading
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    logging.info("Loaded data")

    # Verify if the key already exists
    if (args.from_dpp is not None and 
        args.from_dpp not in sft.data_per_point.keys()):
        parser.error("dpp key not found")
    if (args.from_dps is not None and 
        args.from_dps not in sft.data_per_streamline.keys()):
        parser.error("dps key not found")

    if args.from_dps is not None:
        data = sft.data_per_streamline[args.from_dps]
        data = [np.squeeze(data_s) for data_s in data] 
        if len(data[0]).shape > 1:
            parser.error(
                "Script not ready to deal with dps of more than one value per "
                "streamline. Use scil_tractogram_dps_math to modify your data.")
    else:
        data = sft.data_per_point[args.from_dpp]
        if len(np.squeeze(data[0][0, :]).shape) > 1:
            parser.error(
                "Script not ready to deal with dpp of more than one value per "
                "point. Use scil_tractogram_dpp_math to modify your data.")
        data = [np.mean(np.squeeze(data_s)) for data_s in data] 

    nb_init = len(sft)

    if args.percent or args.nb:
        ordered_ind = np.argsort(data)

        if args.percent:
            nb_streamlines = int(args.percent / 100.0 * len(sft))
            percent = args.percent
        else:
            nb_streamlines = args.nb
            percent = np.round(nb_streamlines / len(sft) * 100, decimals=3)

        if args.top:
            ind = ordered_ind[-nb_streamlines:]
            logging.info("Saving {}/{} streamlines; the top {}% of "
                         "streamlines."
                         .format(len(ind), nb_init, percent))
        elif args.bottom:
            ind = ordered_ind[0:nb_streamlines]
            logging.info("Saving {}/{} streamlines; the bottom {}% of "
                         "streamlines."
                         .format(len(ind), nb_init, percent))
        else:  # args.center
            half_remains = int((len(sft) - nb_streamlines) / 2)
            ind = ordered_ind[half_remains:-half_remains]
            logging.info("Saving {}/{} streamlines; the middle {}% of "
                         "streamlines."
                         .format(len(ind), nb_init, percent))

    else:  # Using mean +- STD
        mean = np.mean(data)
        std = np.std(data)

        if args.top:
            limit = mean + args.std * std
            ind = data >= limit
            logging.info("Number of streamlines above mean + {}std limit: {}"
                         .format(args.std, sum(ind)))
        elif args.bottom:
            limit = mean - args.std * std
            ind = data <= limit
            logging.info("Number of streamlines below mean - {}std limit: {}"
                         .format(args.std, sum(ind)))
        else: # args.center
            limit1 = mean - args.std * std
            limit2 = mean + args.std * std
            ind = np.logical_and(data > limit1, data < limit2)
            logging.info("Number of streamlines in the range mean +- {}std: {}"
                         .format(args.std, sum(ind)))

    sft = sft[ind]
    save_tractogram(sft, args.out_tractogram, no_empty=args.no_empty)


if __name__ == "__main__":
    main()
