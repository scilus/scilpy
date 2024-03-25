#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projects maps extracted from a map onto the points of streamlines.

The default options will take data from a nifti image (3D or 4D) and
project it onto the points of streamlines. If the image is 4D, the data
is stored as a list of 1D arrays per streamline. If the image is 3D,
the data is stored as a list of values per streamline.

See also scil_tractogram_project_streamlines_to_map.py for the reverse action.

* Note that the data from your maps will be projected only on the coordinates
of the points of your streamlines. Data underlying the whole segments between
two consecutive points is not used. If your streamlines are strongly
compressed, or if they have a very big step size, the result will possibly
reflect poorly your map. You may use scil_tractogram_resample.py to upsample
your streamlines first.
* Hint: The streamlines themselves are not modified here, only their dpp. To
avoid multiplying data on disk, you could use the following arguments to save
the new dpp in your current tractogram:
>> scil_tractogram_project_map_to_streamlines.py $in_bundle $in_bundle
       --keep_all_dpp -f
"""

import argparse
import logging

import nibabel as nib
from dipy.io.streamline import save_tractogram

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
                   'streamlines will be NaN). If not set, will project \n'
                   'the map onto all points of the streamlines.')

    p.add_argument('--keep_all_dpp', action='store_true',
                   help='If set, previous data_per_point will be preserved \n'
                   'in the output tractogram. Else, only --out_dpp_name \n'
                   'keys will be saved.')
    p.add_argument('--overwrite_dpp', action='store_true',
                   help='If set, if --keep_all_dpp is set and some \n'
                   '--out_dpp_name keys already existed in your \n'
                   'data_per_point, allow overwriting old data_per_point.')

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram] + args.in_maps,
                        args.reference)
    assert_outputs_exist(parser, args, [args.out_tractogram])

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    logging.info("Loading the tractogram...")
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_voxmm()
    sft.to_corner()

    if len(sft.streamlines) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(
            args.in_tractogram))
        return

    # Check to see if the number of maps and dpp_names are the same
    if len(args.in_maps) != len(args.out_dpp_name):
        parser.error('The number of maps and dpp_names must be the same.')

    # Check to see if there are duplicates in the out_dpp_names
    if len(args.out_dpp_name) != len(set(args.out_dpp_name)):
        parser.error('The output names (out_dpp_names) must be unique.')

    # Check to see if the output names already exist in the input tractogram
    if not args.overwrite_dpp:
        for out_dpp_name in args.out_dpp_name:
            if out_dpp_name in sft.data_per_point:
                logging.info('out_name {} already exists in input tractogram. '
                             'Set overwrite_data or choose a different '
                             'out_name. Exiting.'.format(out_dpp_name))
                return

    data_per_point = {}
    for fmap, dpp_name in zip(args.in_maps, args.out_dpp_name):
        logging.info("Loading the map...")
        map_img = nib.load(fmap)
        map_data = map_img.get_fdata(caching='unchanged', dtype=float)
        map_res = map_img.header.get_zooms()[:3]

        if args.trilinear:
            interp = "trilinear"
        else:
            interp = "nearest"

        map_volume = DataVolume(map_data, map_res, interp)

        logging.info("Projecting map onto streamlines")
        streamline_data = project_map_to_streamlines(
            sft, map_volume,
            endpoints_only=args.endpoints_only)

        logging.info("Saving the tractogram...")

        data_per_point[dpp_name] = streamline_data

    if args.keep_all_dpp:
        sft.data_per_point.update(data_per_point)
        out_sft = sft
    else:
        out_sft = sft.from_sft(sft.streamlines, sft,
                               data_per_point=data_per_point)

    print("New data_per_point keys are: ")
    for key in args.out_dpp_name:
        print("  - {} with shape per point {}"
              .format(key, out_sft.data_per_point[key][0].shape[1:]))

    save_tractogram(out_sft, args.out_tractogram)


if __name__ == '__main__':
    main()
