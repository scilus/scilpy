#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize bundles.

Example usages:

# Visualize streamlines as tubes, each bundle with a different color
>>> scil_viz_bundle.py path_to_bundles/ --shape tube --random_coloring 1337

# Visualize a tractogram with each streamlines drawn as lines, colored with
# their local orientation, but only load 1 in 10 streamlines
>>> scil_viz_bundle.py tractogram.trk --shape line --subsample 10

# Visualize CSTs as large tubes and color them from a list of colors in a file
>>> scil_viz_bundle.py path_to_bundles/CST_* --width 0.5
    --color_dict colors.json
"""

import argparse
import colorsys
import glob
import json
import itertools
import logging
import nibabel as nib
import numpy as np
import os
import random

from dipy.tracking.streamline import set_number_of_points
from fury import window, actor, colormap

from scilpy.io.utils import (assert_inputs_exist,
                             add_verbose_arg,
                             parser_color_type)


streamline_actor = {'tube': actor.streamtube,
                    'line': actor.line}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundles', nargs='+',
                   help='List of tractography files supported by nibabel.')
    p2 = p.add_argument_group(title='Colouring options')
    coloring_group = p2.add_mutually_exclusive_group()
    coloring_group.add_argument('--random_coloring', metavar='SEED', type=int,
                                help='Assign a random color to bundles.')
    coloring_group.add_argument('--uniform_coloring', metavar=('R', 'G', 'B'),
                                nargs=3, type=int,
                                help='Assign a uniform color to streamlines.')
    coloring_group.add_argument('--local_coloring', action='store_true',
                                help='Assign coloring to streamlines '
                                'depending on their local orientations.')
    coloring_group.add_argument('--color_dict', type=str, metavar='JSON',
                                help='JSON file containing colors for each '
                                'bundle.\nBundle filenames are indicated as '
                                ' keys and colors as values.\nA \'default\' '
                                ' key and value can be included.')
    coloring_group.add_argument('--color_from_streamlines',
                                metavar='KEY', type=str,
                                help='Extract a color per streamline from the '
                                'data_per_streamline property of the '
                                'tractogram at the specified key.')
    coloring_group.add_argument('--color_from_points', metavar='KEY', type=str,
                                help='Extract a color per point from the '
                                'data_per_point property of the tractogram '
                                'at the specified key.')
    p.add_argument('--shape', type=str,
                   choices=['line', 'tube'], default='tube',
                   help='Display streamlines either as lines or tubes.'
                   '\n[Default: %(default)s]')
    p.add_argument('--width', type=float, default=0.25,
                   help='Width of tubes or lines representing streamlines'
                   '\n[Default: %(default)s]')
    p.add_argument('--subsample', type=int, default=1,
                   help='Only load 1 in N streamlines.'
                   '\n[Default: %(default)s]')
    p.add_argument('--downsample', type=int, default=None,
                   help='Downsample streamlines to N points.'
                   '\n[Default: %(default)s]')
    p.add_argument('--background', metavar=('R', 'G', 'B'), nargs=3,
                   default=[0, 0, 0], type=parser_color_type,
                   help='RBG values [0, 255] of the color of the background.'
                   '\n[Default: %(default)s]')

    add_verbose_arg(p)

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
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

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

    assert_inputs_exist(parser, bundle_filenames, args.color_dict)

    scene = window.Scene()
    scene.background(tuple(map(int, args.background)))

    def subsample(list_obj):
        """ Lazily subsample a list
        """
        return list(
            itertools.islice(list_obj, 0, None, args.subsample))

    # Load each bundle, subsample and downsample it if needed
    for filename in bundle_filenames:
        try:
            # Lazy-load streamlines to minimize ram usage
            tractogram_gen = nib.streamlines.load(
                filename, lazy_load=True).tractogram
            streamlines_gen = tractogram_gen.streamlines
        except ValueError:
            # Not a file loadable by nibabel's streamline API
            print('Skipping {}'.format(filename))
            continue

        # Actually load streamlines according to the subsample argument
        streamlines = subsample(streamlines_gen)

        if args.downsample:
            streamlines = set_number_of_points(streamlines, args.downsample)

        # Handle bundle colors. Either assign a random bright color to each
        # bundle, or load a color specific to each bundle, or let the bundles
        # be colored according to their local orientation
        if args.random_coloring:
            color = random_rgb()
        elif args.color_dict:
            with open(args.color_dict) as json_file:
                # Color dictionary
                color_dict = json.load(json_file)

                # Extract filenames to compare against the color dictionary
                basename = os.path.splitext(os.path.basename(filename))[0]

                # Load colors
                color = color_dict[basename] \
                    if basename in color_dict.keys() \
                    else color_dict['default']
        elif args.color_from_streamlines:
            color = subsample(
                tractogram_gen.data_per_streamline[args.color_from_streamlines]
            )
        elif args.color_from_points:
            color = subsample(
                tractogram_gen.data_per_point[args.color_from_points])
        elif args.uniform_coloring:  # Assign uniform coloring to streamlines
            color = tuple(np.asarray(args.uniform_coloring) / 255)
        elif args.local_coloring:  # Compute coloring from local orientations
            # Compute segment orientation
            diff = [np.diff(list(s), axis=0) for s in streamlines]
            # Repeat first segment so that the number of segments matches
            # the number of points
            diff = [[d[0]] + list(d) for d in diff]
            # Flatten the list of segments
            orientations = np.asarray([o for d in diff for o in d])
            # Turn the segments into colors
            color = colormap.orient2rgb(orientations)
        else:  # Streamline color will depend on the streamlines' endpoints.
            color = None
        # TODO: Coloring from a volume of local orientations

        line_actor = streamline_actor[args.shape](
            streamlines, colors=color, linewidth=args.width)
        scene.add(line_actor)

    # If there's actually streamlines to display
    if len(bundle_filenames):
        # Showtime !
        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()


if __name__ == '__main__':
    main()
