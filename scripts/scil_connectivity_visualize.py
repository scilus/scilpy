#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to display a connectivity matrix and adjust the desired visualization.
Made to work with scil_tractogram_segment_bundles_for_connectivity.py and
scil_connectivity_reorder_rois.py.

This script can either display the axis labels as:
- Coordinates (0..N)
- Labels (using --labels_list)
- Names (using --labels_list and --lookup_table)
Examples of labels_list.txt and lookup_table.json can be found in the
freesurfer_flow output (https://github.com/scilus/freesurfer_flow)

If the matrix was made from a bigger matrix using
scil_connectivity_reorder_rois.py, provide the text file(s), using
--labels_list and/or --reorder_txt.

The chord chart is always displaying parting in the order they are defined
(clockwise), the color is attributed in that order following a colormap. The
thickness of the line represent the 'size/intensity', the greater the value is
the thicker the line will be. In order to hide the low values, two options are
available:
- Angle threshold + alpha, any connections with a small angle on the chord
    chart will be slightly transparent to increase the focus on bigger
    connections.
- Percentile, hide any connections with a value below that percentile
"""

import argparse
import copy
import json
import math
import logging

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from scilpy.image.volume_math import EPSILON
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_verbose_arg,
                             load_matrix_in_any_format)
from scilpy.viz.legacy.chord_chart import chordDiagram, polar2xy
from scilpy.viz.color import get_lookup_table


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_matrix',
                   help='Connectivity matrix in numpy (.npy) format.')
    p.add_argument('out_png',
                   help='Output filename for the connectivity matrix figure.')

    g1 = p.add_argument_group(title='Naming options')
    g1.add_argument('--labels_list',
                    help='List saved by the decomposition script,\n'
                         'must contain labels rather than coordinates (.txt).')
    g1.add_argument('--reorder_txt',
                    help='File with two rows (x/y) listing the ordering '
                         '(.txt).')
    g1.add_argument('--lookup_table',
                    help='Lookup table with the label number as keys and the '
                         'name as values (.json).')

    g2 = p.add_argument_group(title='Matplotlib options')
    g2.add_argument('--name_axis', action='store_true',
                    help='Use the provided info/files to name axis.')
    g2.add_argument('--axis_text_size', nargs=2, metavar=('X_SIZE', 'Y_SIZE'),
                    default=(10, 10),
                    help='Font size of the X and Y axis labels. [%(default)s]')
    g2.add_argument('--axis_text_angle', nargs=2,
                    metavar=('X_ANGLE', 'Y_ANGLE'),
                    default=(90, 0),
                    help='Text angle of the X and Y axis labels. '
                         '[%(default)s]')
    g2.add_argument('--colormap', default='viridis',
                    help='Colormap to use for the matrix. [%(default)s]')
    g2.add_argument('--display_legend', action='store_true',
                    help='Display the colorbar next to the matrix.')
    g2.add_argument('--legend_min_max', nargs=2, metavar=('MIN', 'MAX'),
                    type=float, default=None,
                    help='Manually define the min/max of the legend.')
    g2.add_argument('--write_values', nargs=2, metavar=('FONT_SIZE',
                                                        'DECIMAL'),
                    default=None, type=int,
                    help='Write the values at the center of each node.\n'
                         'The font size and the rouding parameters can be '
                         'adjusted.')

    histo = p.add_argument_group(title='Histogram options')
    histo.add_argument('--histogram', metavar='FILENAME',
                       help='Compute and display/save an histogram of weights.'
                       )
    histo.add_argument('--nb_bins', type=int,
                       help='Number of bins to use for the histogram.')
    histo.add_argument('--exclude_zeros', action='store_true',
                       help='Exclude the zeros from the histogram.')

    chord = p.add_argument_group(title='Chord chart options')
    chord.add_argument('--chord_chart', metavar='FILENAME',
                       help='Compute and display/save a chord chart of weigth.'
                       )
    chord.add_argument('--percentile_threshold', type=int, default=0,
                       help='Discard connections below that percentile.'
                            '[%(default)s]')
    chord.add_argument('--angle_threshold', type=float, default=1,
                       help='Angle below that theshold will be transparent.\n'
                            'Use --alpha to set opacity. Value typically'
                            'between 0.1 and 5 degrees. [%(default)s]')
    chord.add_argument('--alpha', type=float, default=0.9,
                       help='Opacity for the smaller angle on the chord (0-1).'
                            ' [%(default)s]')
    chord.add_argument('--text_size', default=10, type=float,
                       help='Size of the font for the parcels name/number '
                            '[%(default)s].')
    chord.add_argument('--text_distance', type=float, default=1.1,
                       help='Distance from the center so the parcels '
                            'name/number do not overlap \nwith the diagram '
                            '[%(default)s].')

    p.add_argument('--log', action='store_true',
                   help='Apply a base 10 logarithm to the matrix.')
    p.add_argument('--show_only', action='store_true',
                   help='Do not save the figure, simply display it.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def write_values(ax, matrix, properties):
    width, height = matrix.shape
    font0 = FontProperties().copy()
    font0.set_size(properties[0])
    for x in range(width):
        for y in range(height):
            value = round(matrix[x][y], properties[1])
            ax.annotate(str(value), xy=(x, y), fontproperties=font0,
                        horizontalalignment='center')
    return ax


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_matrix)
    if not args.show_only:
        assert_outputs_exist(parser, args, args.out_png, args.histogram)

    if args.lookup_table and not args.labels_list:
        parser.error('Lookup table axis naming requires --labels_list.')

    matrix = load_matrix_in_any_format(args.in_matrix)
    matrix[np.isnan(matrix)] = 0
    if args.log:
        matrix[matrix > EPSILON] = np.log10(matrix[matrix > EPSILON])
        min_value = np.min(matrix)
        matrix[np.abs(matrix) < EPSILON] = -65536
    else:
        min_value = np.min(matrix)

    max_value = None
    if args.legend_min_max is not None:
        min_value = args.legend_min_max[0]
        max_value = args.legend_min_max[1]

    cmap = get_lookup_table(args.colormap)
    fig, ax = plt.subplots()
    im = ax.imshow(matrix.T,
                   interpolation='nearest',
                   cmap=cmap, vmin=min_value, vmax=max_value)

    if args.write_values:
        if np.prod(matrix.shape) > 1000:
            logging.warning('Large matrix, please consider not using '
                            '--write_values.')
        ax = write_values(ax, matrix, args.write_values)

    if args.display_legend:
        fig.colorbar(im, ax=ax)

    if args.name_axis:
        y_ticks = np.arange(matrix.shape[1])
        x_ticks = np.arange(matrix.shape[0])

        if args.labels_list:
            labels_list = np.loadtxt(args.labels_list, dtype=np.int16).tolist()

        if args.labels_list and not args.reorder_txt and not args.lookup_table:
            if len(labels_list) != matrix.shape[0] \
                    or len(labels_list) != matrix.shape[1]:
                logging.warning('The provided matrix not the same size as '
                                'the labels list.')
            y_legend = labels_list[0:matrix.shape[1]]
            x_legend = labels_list[0:matrix.shape[0]]
        else:
            y_legend = y_ticks
            x_legend = x_ticks

        if args.reorder_txt:
            with open(args.reorder_txt, 'r') as my_file:
                lines = my_file.readlines()
                y_legend = [int(val) for val in lines[1].split()]
                x_legend = [int(val) for val in lines[0].split()]

        if args.lookup_table:
            if args.reorder_txt:
                logging.warning('Using a lookup table, make sure the '
                                'reordering json contain labels, not '
                                'coordinates')
            with open(args.lookup_table) as json_data:
                lut = json.load(json_data)

            y_legend = []
            x_legend = []
            if args.reorder_txt:
                with open(args.reorder_txt, 'r') as my_file:
                    lines = my_file.readlines()
                    y_list = [int(val) for val in lines[1].split()]
                    x_list = [int(val) for val in lines[0].split()]
            else:
                y_list = labels_list[0:matrix.shape[1]]
                x_list = labels_list[0:matrix.shape[0]]

            y_legend = [lut[str(x)] if str(x) in lut else str(x)
                        for x in y_list]
            x_legend = [lut[str(x)] if str(x) in lut else str(x)
                        for x in x_list]

        if len(y_ticks) != len(y_legend) \
                or len(x_ticks) != len(x_legend):
            logging.warning('Legend is not the same size as the data.'
                            'Make sure you are using the same reordering '
                            'json.')

        plt.xticks(x_ticks, x_legend,
                   rotation=int(args.axis_text_angle[0]),
                   ha='right',
                   fontsize=args.axis_text_size[0])
        plt.yticks(y_ticks, y_legend,
                   rotation=int(args.axis_text_angle[1]),
                   fontsize=args.axis_text_size[1])

    if args.show_only:
        plt.show()
    else:
        plt.savefig(args.out_png, dpi=300, bbox_inches='tight')

    if args.histogram:
        fig, ax = plt.subplots()
        if args.exclude_zeros:
            matrix_hist = matrix[matrix != 0]
        else:
            matrix_hist = matrix.ravel()

        _, _, patches = ax.hist(matrix_hist, bins=args.nb_bins)
        nbr_bins = len(patches)
        color = get_lookup_table(args.colormap)(np.linspace(0, 1, nbr_bins))
        for i in range(0, nbr_bins):
            patches[i].set_facecolor(color[i])

        if args.show_only:
            plt.show()
        else:
            plt.savefig(args.histogram, dpi=300, bbox_inches='tight')

    if args.chord_chart:
        if not args.name_axis:
            if matrix.shape[0] != matrix.shape[1]:
                print('Warning, the matrix is not square, the parcels order on'
                      'both axis must be the same.')
            x_legend = [str(i) for i in range(matrix.shape[0])]
            y_legend = [str(i) for i in range(matrix.shape[1])]
        if isinstance(x_legend, np.ndarray):
            x_legend = x_legend.tolist()
            y_legend = y_legend.tolist()

        total_legend = copy.copy(x_legend)
        total_legend.extend(y_legend)
        total_legend = set(total_legend)
        if args.lookup_table:
            total_legend = sorted(total_legend)
        else:
            total_legend = sorted(total_legend, key=int)

        new_matrix = np.zeros((len(total_legend), len(total_legend)))
        for x in range(len(total_legend)):
            for y in range(len(total_legend)):
                if total_legend[x] in x_legend and total_legend[y] in y_legend:
                    i = x_legend.index(total_legend[x])
                    j = y_legend.index(total_legend[y])
                    new_matrix[x, y] = matrix[i, j]
                    new_matrix[y, x] = matrix[i, j]

        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes([0, 0, 1, 1])
        new_matrix[new_matrix < np.percentile(new_matrix,
                                              args.percentile_threshold)] = 0

        empty_to_del = (np.where(~new_matrix.any(axis=1))[0])
        non_empty_to_keep = np.setdiff1d(range(len(total_legend)),
                                         empty_to_del)
        total_legend = [total_legend[i] for i in non_empty_to_keep]
        new_matrix = np.delete(new_matrix, empty_to_del, axis=0)
        new_matrix = np.delete(new_matrix, empty_to_del, axis=1)

        colors = [cmap(i)[0:3] for i in np.linspace(0, 1, len(new_matrix))]
        nodePos = chordDiagram(new_matrix, ax, colors=colors,
                               angle_threshold=args.angle_threshold,
                               alpha=args.alpha, text_dist=args.text_distance)
        ax.axis('off')
        prop = dict(fontsize=args.text_size, ha='center', va='center')
        previous_val = 0
        first_flip = False
        flip = 1
        for i in range(len(new_matrix)):
            radians = math.radians(nodePos[i][2])
            if nodePos[i][2] > previous_val:
                previous_val = nodePos[i][2]
            else:
                flip = -1
                previous_val = 0
                first_flip = True

            if nodePos[i][2] > 270:
                flip = 1 if first_flip else -1
            if isinstance(total_legend[i], str):
                text_len = len(total_legend[i])
            else:
                text_len = len(str(total_legend[i]))

            textPos = polar2xy(text_len*args.text_size*0.001*flip, radians)

            ax.text(nodePos[i][0] + textPos[0], nodePos[i][1]+textPos[1],
                    total_legend[i], rotation=nodePos[i][2], **prop)

        if args.show_only:
            plt.show()
        else:
            plt.savefig(args.chord_chart, dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    main()
