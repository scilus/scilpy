#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The script uses scalars from an anatomy, data_per_point or data_per_streamline
(e.g commit_weights) to visualize them on the streamlines.
Saves the RGB values in the data_per_point (color_x, color_y, color_z).

If called with .tck, the output will always be .trk, because data_per_point has
no equivalent in tck file.

The usage of --use_dps, --use_dpp and --from_anatomy is more complex. It maps
the raw values from these sources to RGB using a colormap.
    --use_dps: total nbr of streamlines of the tractogram = len(streamlines)
    --use_dpp: total nbr of points of the tractogram = len(streamlines._data)

A minimum and a maximum range can be provided to clip values. If the range of
values is too large for intuitive visualization, a log transform can be
applied.

If the data provided from --use_dps, --use_dpp and --from_anatomy are integer
labels, they can be mapped using a LookUp Table (--LUT).
The file provided as a LUT should be either .txt or .npy and if the
size is N=20, then the data provided should be between 1-20.

Example: Use --from_anatomy with a voxel labels map (values from 1-20) with a
text file containing 20 p-values to map p-values to the bundle for
visualisation.

The colormap used for mapping values to colors can be saved to a png/jpg image
using the --out_colorbar option.
"""

import argparse
import logging

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from scilpy.io.streamlines import load_tractogram_with_reference, save_tractogram
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_reference_arg,
                             load_matrix_in_any_format)

COLORBAR_NB_VALUES = 255


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Input tractogram (.trk or .tck).')
    p.add_argument('out_tractogram',
                   help='Output tractogram (.trk or .tck).')

    cbar_g = p.add_argument_group('Colorbar Options')
    cbar_g.add_argument('--out_colorbar',
                        help='Optional output colorbar (.png, .jpg or any '
                             'format supported by matplotlib).')
    cbar_g.add_argument('--horizontal_cbar', action='store_true',
                        help='Draw horizontal colorbar (vertical by default).')

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
        LUT = load_matrix_in_any_format(args.LUT)
        for i, val in enumerate(LUT):
            data[data == i+1] = val

    if args.min_range is not None or args.max_range is not None:
        data = np.clip(data, args.min_range, args.max_range)

    # get data values range
    lbound = np.min(data)
    ubound = np.max(data)

    if args.log:
        data[data > 0] = np.log10(data[data > 0])

    # normalize data between 0 and 1
    data -= np.min(data)
    data = data / np.max(data) if np.max(data) > 0 else data
    return data, lbound, ubound


def save_colorbar(cmap, lbound, ubound, args):
    gradient = cmap(np.linspace(0, 1, COLORBAR_NB_VALUES))[:, 0:3]

    # TODO: Is there a better way to draw a gradient-filled rectangle?
    width = int(COLORBAR_NB_VALUES * 0.1)
    gradient = np.tile(gradient, (width, 1, 1))
    if not args.horizontal_cbar:
        gradient = np.swapaxes(gradient, 0, 1)

    _, ax = plt.subplots(1, 1)
    ax.imshow(gradient, origin='lower')

    ticks_labels = ['{0:.3f}'.format(lbound), '{0:.3f}'.format(ubound)]

    if args.log:
        ticks_labels[0] = 'log(' + ticks_labels[0] + ')'
        ticks_labels[1] = 'log(' + ticks_labels[1] + ')'

    if not args.horizontal_cbar:
        ax.set_yticks([0, COLORBAR_NB_VALUES - 1])
        ax.set_yticklabels(ticks_labels)
        ax.set_xticks([])
    else:
        ax.set_xticks([0, COLORBAR_NB_VALUES - 1])
        ax.set_xticklabels(ticks_labels)
        ax.set_yticks([])

    plt.savefig(args.out_colorbar, bbox_inches='tight')


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram,
                         optional=args.out_colorbar)

    if args.horizontal_cbar and not args.out_colorbar:
        logging.warning('Colorbar output not supplied. Ignoring '
                        '--horizontal_cbar.')

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    if args.LUT:
        LUT = load_matrix_in_any_format(args.LUT)
        if np.any(sft.streamlines._lengths < len(LUT)):
            logging.warning('Some streamlines have fewer point than the size '
                            'of the provided LUT.\nConsider using '
                            'scil_resample_streamlines.py')

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
        data, lbound, ubound = transform_data(args, data)
        color = cmap(data)[:, 0:3] * 255
    elif args.from_anatomy:
        data = nib.load(args.from_anatomy).get_fdata()
        data, lbound, ubound = transform_data(args, data)

        sft.to_vox()
        values = map_coordinates(data, sft.streamlines._data.T,
                                 order=0)
        color = cmap(values)[:, 0:3] * 255
        sft.to_rasmm()
    else:
        parser.error('No coloring method specified.')

    if len(color) == len(sft):
        tmp = [np.tile([color[i][0], color[i][1], color[i][2]],
                       (len(sft.streamlines[i]), 1))
               for i in range(len(sft.streamlines))]
        sft.data_per_point['color'] = tmp
    elif len(color) == len(sft.streamlines._data):
        sft.data_per_point['color'] = sft.streamlines
        sft.data_per_point['color']._data = color
    save_tractogram(sft, args.out_tractogram)

    # output colormap
    if args.out_colorbar:
        save_colorbar(cmap, lbound, ubound, args)


if __name__ == '__main__':
    main()
