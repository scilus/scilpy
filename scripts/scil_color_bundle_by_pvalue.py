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

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, parser_color_type)
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

streamline_actor = {'tube': actor.streamtube,
                    'line': actor.line}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Input bundle file.')
    p.add_argument('in_pvalues',
                   help='Input p-values text file of '
                        'whitespace-separated values.')
    p.add_argument('out_tractogram',
                   help='Output tractogram file (.trk).')
    p.add_argument('out_colormap',
                   help='Output colormap image.')

    p.add_argument('--colormap', default='plasma',
                   help='Colormap to use for coloring the streamlines.')
    p.add_argument('--resample', type=int,
                   help='Optionally resample the streamlines.')

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

    add_overwrite_arg(p)
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

    assert_inputs_exist(parser, [args.in_bundle, args.in_pvalues])
    assert_outputs_exist(parser, args,
                         [args.out_tractogram, args.out_colormap])

    pvalues = np.loadtxt(args.in_pvalues)
    n_timepoints = len(pvalues)

    tractogram = nib.streamlines.load(args.in_bundle)
    streamlines = tractogram.streamlines

    # create interpolation function
    f_interp = interp1d(np.linspace(0.0, 1.0, n_timepoints), pvalues)

    # interpolation weights [0, 1) for p-values
    mapped_pvals = []
    resampled_strl = []
    for strl in streamlines:
        if len(strl) < n_timepoints or args.resample is not None:
            resample = args.resample if args.resample is not None\
                 else n_timepoints
            strl = set_number_of_points(strl, resample)
        resampled_strl.append(strl)

        # compute interpolation weights
        distances = np.sqrt(np.sum((strl[1:] - strl[:-1])**2, axis=-1))
        distances = np.append([0], distances)
        weights = np.cumsum(distances)
        weights /= weights[-1]
        pvals = f_interp(weights)
        mapped_pvals.append(pvals)

    # assign colors to streamlines
    color = colormap.create_colormap(np.ravel(mapped_pvals),
                                     args.colormap, auto=False)
    sft = StatefulTractogram(resampled_strl, tractogram, Space.RASMM)
    sft.data_per_point['color'] = sft.streamlines
    sft.data_per_point['color']._data = color * 255
    save_tractogram(sft, args.out_tractogram)

    # output colormap to png file
    xticks = np.linspace(0, 1, 256)
    gradient = colormap.create_colormap(xticks,
                                        args.colormap,
                                        auto=False)
    gradient = gradient[None, ...]  # 2D RGB image
    _, ax = plt.subplots(1, 1, figsize=(10, 1), dpi=100)
    ax.imshow(gradient, aspect=10)
    ax.set_xticks([0, 255])
    ax.set_xticklabels(['0.0', '1.0'])
    ax.set_yticks([])
    plt.savefig(args.out_colormap)

    if args.interactive:
        width = get_width(args)
        scene = window.Scene()
        scene.background(np.asarray(args.background)/255.0)
        line_actor = streamline_actor[args.shape](resampled_strl, colors=color,
                                                  linewidth=width)
        start_pos = np.array([s[0] for s in resampled_strl])
        dot_actor = actor.dots(start_pos)
        scene.add(line_actor)
        scene.add(dot_actor)

        # Showtime !
        showm = window.ShowManager(scene)
        showm.initialize()
        showm.start()


if __name__ == '__main__':
    main()
