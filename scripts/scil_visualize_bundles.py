#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import nibabel as nib
import os
import random
import itertools

import numpy as np

from fury import window, actor

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundles',
                   help='List of tractography files supported by nibabel.')
    coloring_group = p.add_mutually_exclusive_group()
    coloring_group.add_argument('--random_coloring', metavar='SEED', type=int,
                                help='Assign a random color to bundles.')
    coloring_group.add_argument('--color_list', type=str, metavar='FILE',
                                help='File containing colors for each bundle.')
    p.add_argument('--shape', type=str,
                   choices=['line', 'tube'], default='tube',
                   help='Display streamlines either as lines or tubes.')
    p.add_argument('--subsample', metavar='N', type=int, default=1,
                   help='Only visualize 1 in N streamlines. This is useful' +
                        ' in case tractograms are too heavy.')

    add_overwrite_arg(p)
    return p


def random_rgb():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return np.array([r, g, b]) / 255.0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isdir(args.in_bundles):
        bundle_filenames = [os.path.join(args.in_bundles, f)
                            for f in os.listdir(args.in_bundles)]
    else:
        bundle_filenames = glob.glob(args.in_bundles)

    # TODO: Use scilpy io verif functions
    if len(bundle_filenames) < 1:
        parser.error('{} : No files found !'.format(args.in_bundles))

    scene = window.Scene()
    for filename in bundle_filenames:
        try:
            streamlines = nib.streamlines.load(
                filename, lazy_load=True).tractogram.streamlines
        except ValueError:
            print('Skipping {}'.format(filename))
            continue

        color = None
        if args.random_coloring:
            np.random.seed(int(args.random_coloring))
            color = random_rgb()

        line_actor = actor.streamtube(
            list(itertools.islice(streamlines, 0, None, args.subsample)),
            colors=color)
        scene.add(line_actor)

    showm = window.ShowManager(scene, reset_camera=True)
    showm.initialize()
    showm.start()


if __name__ == '__main__':
    main()
