#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to display a connectivity matrix and adjust the desired visualization.
Made to work with scil_decompose_connectivity.py and
scil_reorder_connectivity.py.

This script can either display the axis labels as:
- Coordinates (0..N)
- Labels (using --labels_list)
- Names (using --labels_list and --lookup_table)

If the matrix was made from a bigger matrix using scil_reorder_connectivity.py,
provide the json and specify the key (using --reorder_json)
"""

import argparse
import json
import logging

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from scilpy.image.operations import EPSILON
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, load_matrix_in_any_format)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_matrix',
                   help='Connectivity matrix in numpy (.npy) format.')
    p.add_argument('out_png',
                   help='Output filename for the figure.')

    g1 = p.add_argument_group(title='Naming options')
    g1.add_argument('--labels_list',
                    help='List saved by the decomposition script,\n'
                         'the json must contain labels rather than coordinates.')
    g1.add_argument('--reorder_json', nargs=2, metavar=('FILE', 'KEY'),
                    help='Json file with the sub-network as keys and x/y '
                         'lists as value AND the key to use.')
    g1.add_argument('--lookup_table',
                    help='Lookup table with the label number as keys and the '
                         'name as values.')

    g2 = p.add_argument_group(title='Matplotlib options')
    g2.add_argument('--name_axis', action='store_true',
                    help='Use the provided info/files to name axis.')
    g2.add_argument('--axis_text_size', nargs=2, metavar=('X_SIZE', 'Y_SIZE'),
                    default=(10, 10),
                    help='Font size of the X and Y axis labels. [%(default)s]')
    g2.add_argument('--axis_text_angle', nargs=2, metavar=('X_ANGLE', 'Y_ANGLE'),
                    default=(90, 0),
                    help='Text angle of the X and Y axis labels. [%(default)s]')
    g2.add_argument('--colormap', default='viridis',
                    help='Colormap to use for the matrix. [%(default)s]')
    g2.add_argument('--display_legend', action='store_true',
                    help='Display the colorbar next to the matrix.')
    g2.add_argument('--write_values', nargs=2, metavar=('FONT_SIZE', 'DECIMAL'),
                    default=None, type=int,
                    help='Write the values at the center of each node.\n'
                         'The font size and the rouding parameters can be '
                         'adjusted.')

    histo = p.add_argument_group(title='Histogram options')
    histo.add_argument('--histogram', metavar='FILENAME',
                       help='Compute and display/save an histogram of weigth.')
    histo.add_argument('--nb_bins', type=int,
                       help='Number of bins to use for the histogram.')
    histo.add_argument('--exclude_zeros', action='store_true',
                       help='Exclude the zeros from the histogram.')

    p.add_argument('--log', action='store_true',
                   help='Apply a base 10 logarithm to the matrix.')
    p.add_argument('--show_only', action='store_true',
                   help='Do not save the figure, simply display it.')

    add_overwrite_arg(p)

    return p


def write_values(ax, matrix, properties):
    width, height = matrix.shape
    font0 = FontProperties().copy()
    font0.set_size(properties[0])
    for x in range(width):
        for y in range(height):
            value = round(matrix[y][x], properties[1])
            ax.annotate(str(value), xy=(x, y), fontproperties=font0,
                        horizontalalignment='center')
    return ax


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_matrix)
    if not args.show_only:
        assert_outputs_exist(parser, args, args.out_png, args.histogram)

    matrix = load_matrix_in_any_format(args.in_matrix)

    if args.log:
        matrix[matrix > EPSILON] = np.log10(matrix[matrix > EPSILON])
        min_value = np.min(matrix)
        matrix[np.abs(matrix) < EPSILON] = -65536
    else:
        min_value = np.min(matrix)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix,
                   interpolation='nearest',
                   cmap=args.colormap, vmin=min_value)

    if args.write_values:
        if np.prod(matrix.shape) > 1000:
            logging.warning('Large matrix, please consider not using '
                            '--write_values.')
        ax = write_values(ax, matrix, args.write_values)

    if args.display_legend:
        fig.colorbar(im, ax=ax)

    if args.name_axis:
        x_ticks = np.arange(matrix.shape[0])
        y_ticks = np.arange(matrix.shape[1])

        if args.labels_list:
            labels_list = np.loadtxt(args.labels_list, dtype=np.int16).tolist()

            if not args.reorder_json and not args.lookup_table:
                if len(labels_list) != matrix.shape[0] \
                        or len(labels_list) != matrix.shape[1]:
                    logging.warning('The provided matrix not the same size as '
                                    'the labels list.')
                x_legend = labels_list[0:matrix.shape[0]]
                y_legend = labels_list[0:matrix.shape[1]]

            if args.reorder_json:
                filename, key = args.reorder_json
                with open(filename) as json_data:
                    config = json.load(json_data)

                    x_legend = config[key][0]
                    y_legend = config[key][1]

            if args.lookup_table:
                logging.warning('Using a lookup table, make sure the reordering '
                                'json contain labels, not coordinates')
                with open(args.lookup_table) as json_data:
                    lut = json.load(json_data)

                x_legend = []
                y_legend = []
                if args.reorder_json:
                    x_list = config[key][0]
                    y_list = config[key][1]
                else:
                    x_list = labels_list[0:matrix.shape[0]]
                    y_list = labels_list[0:matrix.shape[1]]

                x_legend = [lut[str(x)] if str(x) in lut else str(x)
                            for x in x_list]
                y_legend = [lut[str(x)] if str(x) in lut else str(x)
                            for x in y_list]

        else:
            x_legend = x_ticks
            y_legend = y_ticks

        if len(x_ticks) != len(x_legend) \
                or len(y_ticks) != len(y_legend):
            logging.warning('Legend is not the same size as the data.'
                            'Make sure you are using the same reordering json.')
        plt.xticks(x_ticks, x_legend,
                   rotation=args.axis_text_angle[0],
                   fontsize=args.axis_text_size[0])
        plt.yticks(y_ticks, y_legend,
                   rotation=args.axis_text_angle[1],
                   fontsize=args.axis_text_size[1])

    if args.show_only:
        plt.show()
    else:
        plt.savefig(args.out_png, dpi=300, bbox_inches='tight')

    if args.histogram:
        fig, ax = plt.subplots()
        if args.exclude_zeros:
            min_value = EPSILON
        N, bins, patches = ax.hist(matrix.ravel(),
                                   range=(min_value, matrix.max()),
                                   bins=args.nb_bins)
        nbr_bins = len(patches)
        color = plt.cm.get_cmap(args.colormap)(np.linspace(0, 1, nbr_bins))
        for i in range(0, nbr_bins):
            patches[i].set_facecolor(color[i])

        if args.show_only:
            plt.show()
        else:
            plt.savefig(args.histogram, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
