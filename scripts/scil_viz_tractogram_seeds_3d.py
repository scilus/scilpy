#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize seeds as 3D points, with heatmaps corresponding to seed density

Example usages:

>>> scil_viz_tractogram_seeds_3d.py seeds.nii.gz --tractogram tractogram.trk
"""

import argparse
import logging
import nibabel as nib
import numpy as np

from fury import window, actor

from scilpy.io.utils import (assert_inputs_exist,
                             add_verbose_arg,
                             parser_color_type)
from scilpy.viz.color import lut_from_matplotlib_name


streamline_actor = {'tube': actor.streamtube,
                    'line': actor.line}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_seed_map',
                   help='Seed density map.')
    p.add_argument('--tractogram', type=str,
                   help='Tractogram coresponding to the seeds.')
    p.add_argument('--colormap', type=str, default='bone',
                   help='Name of the map for the density coloring. Can be'
                   ' any colormap that matplotlib offers.'
                   '\n[Default: %(default)s]')
    p.add_argument('--seed_opacity', type=float, default=0.5,
                   help='Opacity of the contour generated.'
                   '\n[Default: %(default)s]')
    p.add_argument('--tractogram_shape', type=str,
                   choices=['line', 'tube'], default='tube',
                   help='Display streamlines either as lines or tubes.'
                   '\n[Default: %(default)s]')
    p.add_argument('--tractogram_opacity', type=float, default=0.5,
                   help='Opacity of the streamlines.'
                   '\n[Default: %(default)s]')
    p.add_argument('--tractogram_width', type=float, default=0.05,
                   help='Width of tubes or lines representing streamlines.'
                   '\n[Default: %(default)s]')
    p.add_argument('--tractogram_color', metavar='R G B',
                   nargs='+', default=None, type=parser_color_type,
                   help='Color for the tractogram.')
    p.add_argument('--background', metavar='R G B', nargs='+',
                   default=[0, 0, 0], type=parser_color_type,
                   help='RBG values [0, 255] of the color of the background.'
                   '\n[Default: %(default)s]')

    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_seed_map, [args.tractogram])

    # Seed map informations
    seed_map_img = nib.load(args.in_seed_map)
    seed_map_data = seed_map_img.get_fdata().astype(np.uint8)
    seed_map_affine = seed_map_img.affine

    # Load seed density as labels
    values = np.unique(seed_map_data)

    # Create colormap based on labels
    lut = lut_from_matplotlib_name(args.colormap, [values.min(), values.max()],
                                   len(values))

    # Delete 0 from values
    values = np.delete(values, 0)

    colors = np.zeros((len(values), 4))
    for i, v in enumerate(values):
        lut.GetColor(v, colors[i, :3])
        colors[i, 3] = lut.GetOpacity(v)

    scene = window.Scene()
    scene.background(tuple(map(int, args.background)))

    seedroi_actor = actor.contour_from_label(
        seed_map_data, seed_map_affine, color=colors)
    scene.add(seedroi_actor)

    # Load tractogram as tubes or lines, with color if specified
    if args.tractogram:
        tractogram = nib.streamlines.load(args.tractogram).tractogram
        color = None
        if args.tractogram_color:
            color = tuple(map(int, args.tractogram_color))

        line_actor = streamline_actor[args.tractogram_shape](
            tractogram.streamlines,
            opacity=args.tractogram_opacity,
            colors=color,
            linewidth=args.tractogram_width)
        scene.add(line_actor)

    # Showtime !
    showm = window.ShowManager(scene, reset_camera=True)
    showm.initialize()
    showm.start()


if __name__ == '__main__':
    main()
