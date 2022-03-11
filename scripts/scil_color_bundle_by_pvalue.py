#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assign a color to each streamline point based on its p-value. The p-values
are assumed to be in the same order as the streamline points.

The p-values input format is a whitespace-separated list of values saved
in a .txt file.
"""

import argparse
import nibabel as nib
import numpy as np

from dipy.tracking.streamline import set_number_of_points
from fury import window, actor, colormap

from scilpy.io.utils import assert_inputs_exist, parser_color_type
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram

streamline_actor = {'tube': actor.streamtube,
                    'line': actor.line}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Input bundle file.')
    p.add_argument('in_metric',
                   help='Input metric vector as text file of'
                        ' whitespace-separated values.')
    p.add_argument('out_tractogram',
                   help='Output tractogram file.')

    p.add_argument('--colormap', default='plasma',
                   help='Colormap to use for coloring the streamlines.')

    p.add_argument('--interactive', action='store_true',
                   help='If true, fibers will be rendered to a FURY window.')
    p.add_argument('--shape', type=str,
                   choices=['line', 'tube'], default='line',
                   help='Display streamlines either as lines or tubes.'
                   '\n[Default: %(default)s]')
    p.add_argument('--width', type=float,
                   help='Width of tubes or lines representing streamlines'
                   '\n[Default: 2.0 for lines,]')
    p.add_argument('--background', metavar=('R', 'G', 'B'), nargs=3,
                   default=[0, 0, 0], type=parser_color_type,
                   help='RBG values [0, 255] of the color of the background.'
                   '\n[Default: %(default)s]')
    return p


def get_width(args):
    if args.width is None:
        if args.shape == 'line':
            return 2.0
        elif args.shape == 'tube':
            return 0.25
        else:
            raise ValueError('Unknown shape: {}'.format(args.shape))
    else:
        return args.width


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bundle, args.in_metric])

    pvalues = np.loadtxt(args.in_metric)
    n_timepoints = len(pvalues)

    tractogram = nib.streamlines.load(args.in_bundle)
    streamlines = tractogram.streamlines

    # TODO: For each streamline, do not resample. Keep original number of
    # points and interpolate the p-values instead.
    # Or maybe resample only when the number of points is smaller than the
    # number of p-values.
    streamlines = set_number_of_points(streamlines, n_timepoints)

    color = np.tile(pvalues, len(streamlines))
    color = colormap.create_colormap(color, args.colormap, auto=False)

    width = get_width(args)

    sft = StatefulTractogram(streamlines, tractogram, Space.RASMM)
    sft.data_per_point['color'] = sft.streamlines
    sft.data_per_point['color']._data = color * 255
    save_tractogram(sft, args.out_tractogram)

    if args.interactive:
        scene = window.Scene()
        scene.background(np.asarray(args.background)/255.0)
        line_actor = streamline_actor[args.shape](streamlines, colors=color,
                                                  linewidth=width)
        scene.add(line_actor)

        # Showtime !
        showm = window.ShowManager(scene)
        showm.initialize()
        showm.start()


if __name__ == '__main__':
    main()
