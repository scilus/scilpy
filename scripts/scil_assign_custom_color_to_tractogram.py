#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The script use scalar in data_per_point and data_per_streamline
(e.g commit_weights) to visualise on the streamlines.
Saves the RGB values in the data_per_point (color_x, color_y, color_z).

If called with .tck, the output will always be .trk, because data_per_point has
no equivalent in tck file.

The usage of --use_dps, --use_dpp and --from_anatomy is more complex. It maps
the raw values from these sources to RGB using a colormap.

A minimum and a maximum range can be provided to clip values. If the range of
values is too large for intuitive visualization, a log transform can be applied.

If the data provided from --use_dps, --use_dpp and --from_anatomy are integer
labels, they can be mapped using a LookUp Table (--LUT).
The file provided as a LUT should be either .txt or .npy and if the
size is N=20, then the data provided should be between 1-20.

Example: Use --anatomy with a voxel labels map (values from 1-20) with a text
file containing 20 p-values to map p-values to the bundle for visualisation.
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_reference_arg,
                             load_matrix_in_any_format)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Input tractogram (.trk or .tck).')
    p.add_argument('out_tractogram',
                   help='Output tractogram (.trk or .tck).')

    g1 = p.add_argument_group(title='Coloring Methods')
    p1 = g1.add_mutually_exclusive_group()
    p1.add_argument('--use_dps', metavar='DPS_KEY',
                    help='Use the data_per_streamline (scalar) for coloring,\n'
                         'linear scaling from min-max, e.g. commit_weights.')
    p1.add_argument('--use_dpp', metavar='DPP_KEY',
                    help='Use the data_per_point (scalar) for coloring,\n'
                         'linear scaling from min-max.')
    p1.add_argument('--load_dps', metavar='DPS_KEY',
                    help='Load data per streamline (scalar) for coloring')
    p1.add_argument('--load_dpp', metavar='DPP_KEY',
                    help='Load data per point (scalar) for coloring')
    p1.add_argument('--from_anatomy', metavar='FILE',
                    help='Use the voxel data for coloring,\n'
                         'linear scaling from minmax.')

    g2 = p.add_argument_group(title='Coloring Options')
    g2.add_argument('--colormap', default='jet',
                    help='Select the colormap for colored trk (dps/dpp) '
                    '[%(default)s].')
    g2.add_argument('--min_range', type=float,
                    help='Set the minimum value when using dps/dpp/anatomy.')
    g2.add_argument('--max_range', type=float,
                    help='Set the maximum value when using dps/dpp/anatomy.')
    g2.add_argument('--log', action='store_true',
                    help='Apply a base 10 logarithm for colored trk (dps/dpp).')
    g2.add_argument('--LUT', metavar='FILE',
                    help='If the dps/dpp or anatomy contain integer labels, '
                         'the value will be substituted.\nIf the LUT has 20 '
                         'elements, integers from 1-20 in the data will be\n'
                         'replaced by the value in the file (.npy or .txt)')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def transform_data(args, data):
    if args.LUT:
        data = np.round(data).astype(np.int32)
        LUT = load_matrix_in_any_format(args.LUT)
        for i, val in enumerate(LUT):
            data[data == i+1] = val

    if args.min_range is not None or args.max_range:
        data = np.clip(data, args.min_range, args.max_range)
    if args.log:
        data[data > 0] = np.log10(data[data > 0])
    data -= np.min(data)
    data = data / np.max(data) if np.max(data) > 0 else data

    return data


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    cmap = plt.get_cmap(args.colormap)
    if args.use_dps or args.use_dpp or args.load_dps or args.load_dpp:
        if args.use_dps:
            data = np.squeeze(sft.data_per_streamline[args.use_dps])
            # I believe it works well for gaussian distribution, but
            # COMMIT has very weird outliers values
            if args.use_dps == 'commit_weights' \
                    or args.use_dps == 'commit2_weights':
                data = np.clip(data, np.quantile(data, 0.05),
                               np.quantile(data, 0.95))
        elif args.use_dpp:
            data = np.squeeze(sft.data_per_point[args.use_dpp]._data)
        elif args.load_dps:
            data = np.squeeze(load_matrix_in_any_format(args.load_dps))
            if len(data) != len(sft):
                parser.error('Wrong dps size!')
        elif args.load_dpp:
            data = np.squeeze(load_matrix_in_any_format(args.load_dpp))
            if len(data) != len(sft.streamlines._data):
                parser.error('Wrong dpp size!')
        data = transform_data(args, data)
        color = cmap(data)[:, 0:3] * 255
    elif args.from_anatomy:
        data = nib.load(args.from_anatomy).get_fdata()
        data = transform_data(args, data)

        sft.to_vox()
        values = map_coordinates(data, sft.streamlines._data.T,
                                 order=0)
        color = cmap(values)[:, 0:3] * 255
        sft.to_rasmm()

    if len(color) == len(sft):
        tmp = [np.tile([color[i][0], color[i][1], color[i][2]],
                       (len(sft.streamlines[i]), 1))
               for i in range(len(sft.streamlines))]
        sft.data_per_point['color'] = tmp
    elif len(color) == len(sft.streamlines._data):
        sft.data_per_point['color'] = sft.streamlines
        sft.data_per_point['color']._data = color
    save_tractogram(sft, args.out_tractogram)


if __name__ == '__main__':
    main()
