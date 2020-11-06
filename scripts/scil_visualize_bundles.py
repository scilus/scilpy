#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize bundles.

Example usages:

# Visualize streamlines as tubes, each bundle with a different color
$ scil_visualize_bundles path_to_bundles/ --shape tube --random_coloring 1337

# Visualize a tractogram with each streamlines drawn as lines, colored with
# their local orientation, but only load 1 in 10 streamlines
$ scil_visualize_bundles tractogram.trk --shape line --subsample 10

# Visualize CSTs as large tubes and color them from a list of colors in a file
$ scil_visualize_bundles path_to_bundles/CST_* --width 0.5 \
    --color_list colors.txt
"""

import argparse
import colorsys
import glob
import itertools
import nibabel as nib
import numpy as np
import os
import random

from dipy.tracking.streamline import set_number_of_points
from fury import window, actor

from scilpy.io.utils import assert_inputs_exist, parser_color_type


streamline_actor = {'tube': actor.streamtube,
                    'line': actor.line}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundles', nargs='+',
                   help='List of tractography files supported by nibabel.')
    coloring_group = p.add_mutually_exclusive_group()
    coloring_group.add_argument('--random_coloring', metavar='SEED', type=int,
                                help='Assign a random color to bundles.')
    coloring_group.add_argument('--color_list', type=str, metavar='FILE',
                                help='File containing colors for each bundle.'
                                'Color values must be separated by a space'
                                ' and each color must be on its own line.'
                                'Can include opacity.')
    p.add_argument('--shape', type=str,
                   choices=['line', 'tube'], default='tube',
                   help='Display streamlines either as lines or tubes.')
    p.add_argument('--width', type=float, default=0.05,
                   help='Width of tubes or lines representing streamlines')
    p.add_argument('--subsample', metavar='N', type=int, default=1,
                   help='Only load 1 in N streamlines.')
    p.add_argument('--downsample', metavar='N', type=int, default=None,
                   help='Downsample streamlines to N points.')
    p.add_argument('--background', metavar='R G B', nargs='+',
                   default=[0, 0, 0], type=parser_color_type,
                   help='RBG values [0, 255] of the color of the background.')
    return p


def random_rgb():
    # Heuristic to get a random "bright" color
    # From https://stackoverflow.com/a/43437435
    h, s, li = (random.random(),
                0.5 + random.random()/2.0,
                0.4 + random.random()/5.0)
    r, g, b = [int(256*i) for i in colorsys.hls_to_rgb(h, li, s)]
    return np.array([r, g, b])


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Make sure the colors are consistent between executions
    if args.random_coloring is not None:
        random.seed(int(args.random_coloring))

    # Handle bundle filenames. 3 cases are possible:
    # A list of files was passed as arguments
    # A directory was passed as argument
    # A single-file or path containing wildcard was specified
    bundle_filenames = args.in_bundles
    if len(args.in_bundles) == 1:
        # If only one file is specified, it may be a whole folder or
        # a single file, or a wildcard usage
        in_path = args.in_bundles[0]
        if os.path.isdir(in_path):
            # Load the folder
            bundle_filenames = [os.path.join(in_path, f)
                                for f in os.listdir(in_path)]
        else:
            # Load the file/wildcard
            bundle_filenames = glob.glob(in_path)

    assert_inputs_exist(parser, bundle_filenames, args.color_list)

    scene = window.Scene()
    scene.background(tuple(map(int, args.background)))

    # Handle bundle colors. Either assign a random bright color to each
    # bundle, or load a color specific to each bundle, or let the bundles
    # be colored according to their local orientation
    if args.random_coloring is not None:
        colors = [random_rgb() for _ in range(len(bundle_filenames))]
    elif args.color_list:
        colors = map(tuple, np.loadtxt(args.color_list))
    else:
        colors = [None for _ in range(len(bundle_filenames))]

    # Load each bundle, subsample and downsample it if needed and
    # assign it its color. Bundles are sorted alphabetically so their
    # color matches
    for filename, color in zip(sorted(bundle_filenames), colors):
        try:
            # Lazy-load streamlines to minimize ram usage
            streamlines_gen = nib.streamlines.load(
                filename, lazy_load=True).tractogram.streamlines
        except ValueError:
            # Not a file loadable by nibabel's streamline API
            print('Skipping {}'.format(filename))
            continue

        # Actually load streamlines according to the subsample argument
        streamlines = list(
            itertools.islice(streamlines_gen, 0, None, args.subsample))

        if args.downsample:
            streamlines = set_number_of_points(streamlines, args.downsample)

        line_actor = streamline_actor[args.shape](
            streamlines, colors=color, linewidth=args.width)
        scene.add(line_actor)

    # Showtime !
    showm = window.ShowManager(scene, reset_camera=True)
    showm.initialize()
    showm.start()


if __name__ == '__main__':
    main()
