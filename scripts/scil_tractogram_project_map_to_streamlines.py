#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projects metrics extracted from a map onto the endpoints of streamlines.

The default options will take data from a nifti image (3D or ND) and
project it onto the endpoints of streamlines.
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
from scilpy.tractograms.streamline_operations import (
    project_metric_to_streamlines)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    # Mandatory arguments input and output tractogram must be in trk format
    p.add_argument('in_tractogram',
                   help='Fiber bundle file.')
    p.add_argument('out_tractogram',
                   help='Output file.')
    p.add_argument('--in_metric', nargs='+', required=True,
                   help='Nifti metric to project onto streamlines.')
    p.add_argument('--out_dpp_name', nargs='+', required=True,
                   help='Name of the data_per_point to be saved in the \n'
                   'output tractogram.')

    # Optional arguments
    p.add_argument('--trilinear', action='store_true',
                   help='If set, will use trilinear interpolation \n'
                        'else will use nearest neighbor interpolation \n'
                        'by default.')

    p.add_argument('--endpoints_only', action='store_true',
                   help='If set, will only project the metric onto the \n'
                   'endpoints of the streamlines (all other values along \n'
                   ' streamlines will be NaN). If not set, will project \n'
                   ' the metric onto all points of the streamlines.')

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_file_list = [args.in_tractogram]
    for fmetric in args.in_metric:
        input_file_list += fmetric
    assert_inputs_exist(parser, input_file_list)

    assert_outputs_exist(parser, args, [args.out_tractogram])

    logging.debug("Loading the tractogram...")
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_voxmm()
    sft.to_corner()

    if len(sft.streamlines) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return

    # Check to see if the number of metrics and dpp_names are the same
    if len(args.in_metric) != len(args.out_dpp_name):
        parser.error('The number of metrics and dpp_names must be the same.')

    data_per_point = {}
    for fmetric, dpp_name in zip(args.in_metric[0], args.out_dpp_name[0]):
        logging.debug("Loading the metric...")
        metric_img, metric_dtype = load_img(fmetric)
        metric_data = metric_img.get_fdata(caching='unchanged', dtype=float)
        metric_res = metric_img.header.get_zooms()[:3]

        if args.trilinear:
            interp = "trilinear"
        else:
            interp = "nearest"

        metric = DataVolume(metric_data, metric_res, interp)

        logging.debug("Projecting metric onto streamlines")
        streamline_data = project_metric_to_streamlines(
            sft, metric,
            endpoints_only=args.endpoints_only)

        logging.debug("Saving the tractogram...")

        data_per_point[dpp_name] = streamline_data

    out_sft = StatefulTractogram(sft.streamlines, metric_img,
                                 sft.space, sft.origin,
                                 data_per_point=data_per_point)
    save_tractogram(out_sft, args.out_tractogram)


if __name__ == '__main__':
    main()
