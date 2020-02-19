#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description='Flip one or more axes of the '
                                            'encoding scheme matrix.')

    p.add_argument('in_matrix',
                   help='')
    p.add_argument('out_png',
                   help='')

    p.add_argument('--labels_list',
                   help='')
    p.add_argument('--reorder_json', nargs=2, metavar=('FILE', 'KEY'),
                   help='')
    p.add_argument('--lookup_table',
                   help='')

    p.add_argument('--name_axis', action='store_true',
                   help='')
    p.add_argument('--font_size', nargs=2, metavar=('X_SIZE', 'Y_SIZE'), default=(10, 10),
                   help='')
    p.add_argument('--font_angle', nargs=2, metavar=('X_ANGLE', 'Y_ANGLE'), default=(90, 0),
                   help='')

    p.add_argument('--colormap', default='viridis',
                   help='')
    p.add_argument('--show_legend', action='store_true',
                   help='')
    p.add_argument('--write_values', nargs=2, metavar=('FONT_SIZE', 'DECIMAL'), default=None, type=int,
                   help='')

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

    # assert_inputs_exist(parser, args.encoding_file)
    # assert_outputs_exist(parser, args, args.flipped_encoding)

    matrix = np.load(args.in_matrix)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix,
                   interpolation='nearest',
                   cmap=args.colormap,
                   vmin=0.0, vmax=np.max(matrix))

    if args.write_values:
        ax = write_values(ax, matrix, args.write_values)
    if args.show_legend:
        fig.colorbar(im, ax=ax)

    if args.name_axis:
        x_ticks = np.arange(matrix.shape[0])
        y_ticks = np.arange(matrix.shape[1])

        if args.labels_list:
            labels_list = np.loadtxt(args.labels_list).astype(np.int16)

            if not args.reorder_json and not args.lookup_table:
                if len(labels_list) != matrix.shape[0] \
                        or len(labels_list) != matrix.shape[1]:
                    logging.warning('Matrix not the same size as the label list.')
                x_legend = labels_list[0:matrix.shape[0]]
                y_legend = labels_list[0:matrix.shape[1]]

            if args.reorder_json:
                filename, key = args.reorder_json
                with open(filename) as json_data:
                    config = json.load(json_data)

                    x_legend = labels_list[config[key][0]]
                    y_legend = labels_list[config[key][1]]

            if args.lookup_table:
                with open(args.lookup_table) as json_data:
                    lut = json.load(json_data)

                x_legend = []
                y_legend = []
                if args.reorder_json:
                    x_list = labels_list[config[key][0]]
                    y_list = labels_list[config[key][1]]
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
                   rotation=args.font_angle[0], fontsize=args.font_size[0])
        plt.yticks(y_ticks, y_legend,
                   rotation=args.font_angle[1], fontsize=args.font_size[1])
    plt.show()


if __name__ == "__main__":
    main()
