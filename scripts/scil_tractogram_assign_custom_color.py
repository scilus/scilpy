#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The script uses scalars from an anatomy, data_per_point or data_per_streamline
(e.g. commit_weights) to visualize them on the streamlines.
Saves the RGB values in the data_per_point 'color' with 3 values per point:
(color_x, color_y, color_z).

If called with .tck, the output will always be .trk, because data_per_point has
no equivalent in tck file.

If used with a visualization software like MI-Brain
(https://github.com/imeka/mi-brain), the 'color' dps is applied by default at
loading time.

COLORING METHOD
This script maps the raw values from these sources to RGB using a colormap.
    --use_dpp: The data from each point is converted to a color.
    --use_dps: The same color is applied to all points of the streamline.
    --from_anatomy: The voxel's color is used for the points of the streamlines
    crossing it. See also scil_tractogram_project_map_to_streamlines.py. You
    can have more options to project maps to dpp, and then use --use_dpp here.
    --along_profile: The data used here is each point's position in the
    streamline. To have nice results, you should first uniformize head/tail.
    See scil_tractogram_uniformize_endpoints.py.
    --local_angle.

COLORING OPTIONS
A minimum and a maximum range can be provided to clip values. If the range of
values is too large for intuitive visualization, a log transform can be
applied.

If the data provided from --use_dps, --use_dpp and --from_anatomy are integer
labels, they can be mapped using a LookUp Table (--LUT).
The file provided as a LUT should be either .txt or .npy and if the size is
N=20, then the data provided should be between 1-20.

A custom colormap can be provided using --colormap. It should be a string
containing a colormap name OR multiple Matplotlib named colors separated by -.
The colormap used for mapping values to colors can be saved to a png/jpg image
using the --out_colorbar option.

See also: scil_tractogram_assign_uniform_color.py, for simplified options.

Formerly: scil_assign_custom_color_to_tractogram.py
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)
from scilpy.tractograms.dps_and_dpp_management import add_data_as_color_dpp
from scilpy.tractograms.streamline_operations import (get_values_along_length,
                                                      get_angles)
from scilpy.viz.color import get_lookup_table
from scilpy.viz.color import prepare_colorbar_figure


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Input tractogram (.trk or .tck).')
    p.add_argument('out_tractogram',
                   help='Output tractogram (.trk or .tck).')

    cbar_g = p.add_argument_group('Colorbar options')
    cbar_g.add_argument('--out_colorbar',
                        help='Optional output colorbar (.png, .jpg or any '
                             'format \nsupported by matplotlib).')
    cbar_g.add_argument('--show_colorbar', action='store_true',
                        help="Will show the colorbar. Must be used with "
                             "--out_colorbar \nto be effective.")
    cbar_g.add_argument('--horizontal_cbar', action='store_true',
                        help='Draw horizontal colorbar (vertical by default).')

    g1 = p.add_argument_group(title='Coloring method')
    p1 = g1.add_mutually_exclusive_group(required=True)
    p1.add_argument('--use_dps', metavar='DPS_KEY',
                    help='Use the data_per_streamline (scalar) for coloring.')
    p1.add_argument('--use_dpp', metavar='DPP_KEY',
                    help='Use the data_per_point (scalar) for coloring.')
    p1.add_argument('--load_dps', metavar='DPS_FILE',
                    help='Load data per streamline (scalar) for coloring')
    p1.add_argument('--load_dpp', metavar='DPP_FILE',
                    help='Load data per point (scalar) for coloring')
    p1.add_argument('--from_anatomy', metavar='FILE',
                    help='Use the voxel data for coloring,\n'
                         'linear scaling from minmax.')
    p1.add_argument('--along_profile', action='store_true',
                    help='Color streamlines according to each point position'
                         'along its length.')
    p1.add_argument('--local_angle', action='store_true',
                    help="Color streamlines according to the angle between "
                         "each segment (in degree). \nAngles at first and "
                         "last points are set to 0.")

    g2 = p.add_argument_group(title='Coloring options')
    g2.add_argument('--colormap', default='jet',
                    help='Select the colormap for colored trk (dps/dpp) '
                    '[%(default)s].\nUse two Matplotlib named color separeted '
                    'by a - to create your own colormap.')
    g2.add_argument('--clip_outliers', action='store_true',
                    help="If set, we will clip the outliers (first and last "
                         "5%% quantile). Strongly suggested if your data "
                         "comes from COMMIT!")
    g2.add_argument('--min_range', type=float,
                    help='Set the minimum value when using dps/dpp/anatomy.')
    g2.add_argument('--max_range', type=float,
                    help='Set the maximum value when using dps/dpp/anatomy.')
    g2.add_argument('--min_cmap', type=float,
                    help='Set the minimum value of the colormap.')
    g2.add_argument('--max_cmap', type=float,
                    help='Set the maximum value of the colormap.')
    g2.add_argument('--log', action='store_true',
                    help='Apply a base 10 logarithm for colored trk (dps/dpp).'
                    )
    g2.add_argument('--LUT', metavar='FILE',
                    help='If the dps/dpp or anatomy contain integer labels, '
                         'the value will be substituted.\nIf the LUT has 20 '
                         'elements, integers from 1-20 in the data will be\n'
                         'replaced by the value in the file (.npy or .txt)')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram,
                         optional=args.out_colorbar)

    if args.horizontal_cbar and not args.out_colorbar:
        logging.warning('Colorbar output not supplied. Ignoring '
                        '--horizontal_cbar.')

    if (args.use_dps is not None and
            args.use_dps in ['commit_weights', 'commit2_weights'] and not
            args.clip_outliers):
        logging.warning("You seem to be using commit weights. They typically "
                        "have outliers values. We suggest using "
                        "--clip_outliers.")

    # Loading
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    LUT = load_matrix_in_any_format(args.LUT) if args.LUT else None

    cmap = get_lookup_table(args.colormap)

    # Loading data. Depending on the type of loading, format data now to a 1D
    # array (one value per point or per streamline)
    if args.use_dps:
        if args.use_dps not in sft.data_per_streamline.keys():
            parser.error("DPS key {} not found in the loaded tractogram's "
                         "data_per_streamline.".format(args.use_dps))
        data = np.squeeze(sft.data_per_streamline[args.use_dps])
    elif args.use_dpp:
        if args.use_dpp not in sft.data_per_point.keys():
            parser.error("DPP key {} not found in the loaded tractogram's "
                         "data_per_point.".format(args.use_dpp))
        data = np.hstack(
            [np.squeeze(sft.data_per_point[args.use_dpp][s]) for s in
             range(len(sft))])
    elif args.load_dps:
        data = np.squeeze(load_matrix_in_any_format(args.load_dps))
        if len(data) != len(sft):
            parser.error('Wrong dps size! Expected one value per streamline '
                         '({}) but found {} values.'
                         .format(len(sft), len(data)))
    elif args.load_dpp or args.from_anatomy:
        sft.to_vox()
        concat_points = np.vstack(sft.streamlines).T
        expected_shape = len(concat_points)
        sft.to_rasmm()
        if args.load_dpp:
            data = np.squeeze(load_matrix_in_any_format(args.load_dpp))
            if len(data) != expected_shape:
                parser.error('Wrong dpp size! Expected a total of {} points, '
                             'but got {}'.format(expected_shape, len(data)))
        else:  # args.from_anatomy:
            data = nib.load(args.from_anatomy).get_fdata()
            data = map_coordinates(data, concat_points, order=0)
    elif args.along_profile:
        data = get_values_along_length(sft)
    else:  # args.local_angle:
        data = get_angles(sft)

    # Processing
    sft, lbound, ubound = add_data_as_color_dpp(
        sft, cmap, data, args.clip_outliers, args.min_range, args.max_range,
        args.min_cmap, args.max_cmap, args.log, LUT)

    # Saving
    save_tractogram(sft, args.out_tractogram)

    if args.out_colorbar:
        _ = prepare_colorbar_figure(
            cmap, lbound, ubound,
            horizontal=args.horizontal_cbar, log=args.log)
        plt.savefig(args.out_colorbar, bbox_inches='tight')
        if args.show_colorbar:
            plt.show()


if __name__ == '__main__':
    main()
