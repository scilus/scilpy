#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projects maps extracted from a map onto the points of streamlines.

The default options will take data from a nifti image (3D or 4D) and
project it onto the points of streamlines. If the image is 4D, the data
is stored as a list of 1D arrays per streamline. If the image is 3D, the data is stored
as a list of values per streamline.
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram, StatefulTractogram

from scilpy.io.image import load_img
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.image.volume_space_management import DataVolume
from scilpy.tractograms.dps_and_dpp_management import (
    project_map_to_streamlines)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    # Mandatory arguments input and output tractogram must be in trk format
    p.add_argument('in_tractogram',
                   help='Fiber bundle file.')
    p.add_argument('out_tractogram',
                   help='Output file.')
    p.add_argument('--in_maps', nargs='+', required=True,
                   help='Nifti map to project onto streamlines.')
    p.add_argument('--out_dpp_name', nargs='+', required=True,
                   help='Name of the data_per_point to be saved in the \n'
                   'output tractogram.')

    # Optional arguments
    p.add_argument('--trilinear', action='store_true',
                   help='If set, will use trilinear interpolation \n'
                        'else will use nearest neighbor interpolation \n'
                        'by default.')
    p.add_argument('--endpoints_only', action='store_true',
                   help='If set, will only project the map onto the \n'
                   'endpoints of the streamlines (all other values along \n'
                   ' streamlines will be NaN). If not set, will project \n'
                   ' the map onto all points of the streamlines.')
    p.add_argument('--overwrite_data', action='store_true', default=False,
                   help='If set, will overwrite data_per_point in the '
                   'output tractogram, otherwise previous data will be '
                   'preserved in the output tractogram.')

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram] + args.in_maps)

    assert_outputs_exist(parser, args, [args.out_tractogram])

    logging.debug("Loading the tractogram...")
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_voxmm()
    sft.to_corner()

    if len(sft.streamlines) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.in_tractogram))
        return

    # Check to see if the number of maps and dpp_names are the same
    if len(args.in_maps) != len(args.out_dpp_name):
        parser.error('The number of maps and dpp_names must be the same.')

    # Check to see if there are duplicates in the out_dpp_names
    if len(args.out_dpp_name) != len(set(args.out_dpp_name)):
        parser.error('The output names (out_dpp_names) must be unique.')

    # Check to see if the output names already exist in the input tractogram
    if not args.overwrite_data:
        for out_dpp_name in args.out_dpp_name:
            if out_dpp_name in sft.data_per_point:
                logging.info('out_name {} already exists in input tractogram. '
                             'Set overwrite_data or choose a different '
                             'out_name. Exiting.'.format(out_dpp_name))
                return

    data_per_point = {}
    for fmap, dpp_name in zip(args.in_maps, args.out_dpp_name):
        logging.debug("Loading the map...")
        map_img, map_dtype = load_img(fmap)
        map_data = map_img.get_fdata(caching='unchanged', dtype=float)
        map_res = map_img.header.get_zooms()[:3]

        if args.trilinear:
            interp = "trilinear"
        else:
            interp = "nearest"

        map = DataVolume(map_data, map_res, interp)

        logging.debug("Projecting map onto streamlines")
        streamline_data = project_map_to_streamlines(
            sft, map,
            endpoints_only=args.endpoints_only)

        logging.debug("Saving the tractogram...")

        data_per_point[dpp_name] = streamline_data

    if args.overwrite_data:
        out_sft = sft.from_sft(sft.streamlines, sft,
                               data_per_point=data_per_point)
    else:
        old_data_per_point = sft.data_per_point
        for dpp_name in data_per_point:
            old_data_per_point[dpp_name] = data_per_point[dpp_name]
        out_sft = sft.from_sft(sft.streamlines, sft,
                               data_per_point=old_data_per_point)

    save_tractogram(out_sft, args.out_tractogram)


if __name__ == '__main__':
    main()
