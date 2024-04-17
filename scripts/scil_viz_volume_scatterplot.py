#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to display scatter plot between two maps (ex. FA and MD, ihMT and MT).
By default, no mask is applied to the data.
Different options are available to mask or threshold data:
  - a binary mask
  - two probability maps, which can be used to threshold maps with
    --in_prob_maps. A same threshold is applied on these two maps (--thr).
  - parcellation, which can be used to plot values for each region of
    an atlas (--in_atlas) or a subset of regions (--specific_label).
    Atlas option required a json file (--atlas_lut) with indices and
    names of each label corresponding to the atlas as following:
    "1": "lh_A8m",
    "2": "rh_A8m",
    The numbers must be corresponding to the label indices in the json file.

Be careful, you can not use all of them at the same time.

For general scatter plot without mask:
>>> scil_viz_volume_scatterplot.py FA.nii.gz MD.nii.gz out_filename_image.png

For scatter plot with mask:
>>> scil_viz_volume_scatterplot.py FA.nii.gz MD.nii.gz out_filename_image.png
    --in_bin_mask mask_wm.nii.gz

For tissue probability scatter plot:
>>> scil_viz_volume_scatterplot.py FA.nii.gz MD.nii.gz out_filename_image.png
    --prob_maps wm_map.nii.gz gm_map.nii.gz

For scatter plot using atlas:
>>> scil_viz_volume_scatterplot.py FA.nii.gz MD.nii.gz out_filename_image.png
    --in_atlas atlas.nii.gz --atlas_lut atlas.json

>>> scil_viz_volume_scatterplot.py FA.nii.gz MD.nii.gz out_filename_image.png
    --in_atlas atlas.nii.gz --atlas_lut atlas.json
    --specific_label 34 67 87
"""

import argparse
import copy
import json
import logging
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_headers_compatible)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_x_map',
                   help='Map in x axis, FA for example.')
    p.add_argument('in_y_map',
                   help='Map in y axis, MD for example.')
    p.add_argument('out_name',
                   help='Output filename for the figure without extension.')
    p.add_argument('--out_dir',
                   help='Output directory to save scatter plot.')
    p.add_argument('--thr', default=0.9,
                   help='Use to apply threshold only on probability maps '
                        '(same for both map) with --in_prob_maps option.'
                        ' [%(default)s]')
    p.add_argument('--not_exclude_zero', action='store_true',
                   help='Keep zero value in data.')

    maskopt = p.add_mutually_exclusive_group()
    maskopt.add_argument('--in_bin_mask',
                         help='Binary mask. Use this option to extract x and '
                         'y maps value from specific mask or region: '
                         'wm_mask or roi_mask for example.')
    maskopt.add_argument('--in_prob_maps', nargs=2,
                         help='Probability maps, WM and GW for example.')
    maskopt.add_argument('--in_atlas',
                         help='Path to the input atlas image.')

    atlas = p.add_argument_group(title='Atlas options')
    atlas.add_argument('--atlas_lut',
                       help='Path of the LUT file corresponding to atlas '
                            'used to name the regions of interest.')
    atlas.add_argument('--specific_label', type=int, nargs='+', default=None,
                       help='Label list to use to do scatter plot. Label must '
                            'corresponding to atlas LUT file. [%(default)s]')
    atlas.add_argument('--in_folder', action='store_true',
                       help='Save label plots in subfolder "Label_plots".')

    scat = p.add_argument_group(title='Scatter plot options')
    scat.add_argument('--title',
                      default='Scatter Plot',
                      help='Use the provided info for the title name. '
                           ' [%(default)s]')
    scat.add_argument('--x_label', default='x',
                      help='Use the provided info for the x axis name. '
                           ' [%(default)s]')
    scat.add_argument('--y_label', default='y',
                      help='Use the provided info for the y axis name. '
                           ' [%(default)s]')
    scat.add_argument('--label',
                      help='Use the provided info for the legend box '
                           'corresponding to mask or first probability map. '
                           ' [%(default)s]')
    scat.add_argument('--label_prob', default='Threshold prob_map 2',
                      help='Use the provided info for the legend box '
                           'corresponding to the second probability map. '
                           ' [%(default)s]')
    scat.add_argument('--marker', default='.',
                      help='Use the provided info for the marker shape.'
                           ' [%(default)s]')
    scat.add_argument('--marker_size', default=15,
                      help='Use the provided info for the marker size.'
                           ' [%(default)s]')
    scat.add_argument('--transparency', default=0.4,
                      help='Use the provided info for the point transparency.'
                           ' [%(default)s]')
    scat.add_argument('--dpi', default=300,
                      help='Use the provided info for the dpi resolution.'
                           ' [%(default)s]')
    scat.add_argument('--colors', nargs=2, metavar=('color1', 'color2'),
                      default=('r', 'b'))

    p.add_argument('--show_only', action='store_true',
                   help='Do not save the figure, only display. '
                        ' Not avalaible with --in_atlas option.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    maps = [args.in_x_map, args.in_y_map]
    prob_maps = args.in_prob_maps or []
    assert_inputs_exist(parser, maps,
                        optional=prob_maps +
                        [args.in_bin_mask, args.in_atlas, args.atlas_lut])
    assert_headers_compatible(parser, maps,
                              optional=prob_maps +
                              [args.in_bin_mask, args.in_atlas])

    if args.out_dir is None:
        args.out_dir = './'

    # Load x and y images
    maps_data = []
    for curr_map in maps:
        maps_image = nib.load(curr_map)
        if args.not_exclude_zero:
            maps_data.append(maps_image.get_fdata(dtype=np.float32))
        else:
            data = maps_image.get_fdata(dtype=np.float32)
            data[np.where(data == 0)] = np.nan
            maps_data.append(data)

    if args.in_bin_mask:
        if args.label is None:
            args.label = 'Masking data'
        # Load and apply binary mask
        mask_data = get_data_as_mask(nib.load(args.in_bin_mask))
        for curr_map in maps_data:
            curr_map[np.where(mask_data == 0)] = np.nan

    if args.in_prob_maps:
        if args.label is None:
            args.label = 'Threshold prob_map 1'
        # Load tissue probability maps
        prob_masks = []
        for curr_map in args.in_prob_maps:
            prob_image = nib.load(curr_map)
            prob_masks.append(prob_image.get_fdata(dtype=np.float32))

        # Deepcopy to apply the second probability map on same data
        maps_prob = copy.deepcopy(maps_data)

        # Threshold probability images with tissue probability maps
        for curr_map in maps_data:
            curr_map[np.where(prob_masks[0] < args.thr)] = np.nan

        for curr_map in maps_prob:
            curr_map[np.where(prob_masks[1] < args.thr)] = np.nan

    if args.in_atlas:
        label_image = nib.load(args.in_atlas)
        label_data = get_data_as_labels(label_image)

        with open(args.atlas_lut) as f:
            label_dict = json.load(f)
        lut_indices, lut_names = zip(*label_dict.items())

        if args.specific_label:
            label_indices = []
            label_names = []
            for key in args.specific_label:
                label_indices.append(lut_indices[key-1])
                label_names.append(lut_names[key-1])
        else:
            (label_indices, label_names) = (lut_indices, lut_names)

    # Scatter Plots
    # Plot for each label only with unmasking data
    if args.in_atlas:
        if args.in_folder:
            args.out_dir = os.path.join(args.out_dir, 'Label_plots/')
            if not os.path.isdir(args.out_dir):
                os.mkdir(args.out_dir)

        for label, name in zip(label_indices, label_names):
            label = int(label)
            fig, ax = plt.subplots()
            x = (maps_data[0][np.where(label_data == label)])
            y = (maps_data[1][np.where(label_data == label)])

            ax.scatter(x, y, label=name, color=args.colors[0],
                       s=args.marker_size, marker=args.marker,
                       alpha=args.transparency)
            plt.xlabel(args.x_label)
            plt.ylabel(args.y_label)
            plt.title(args.title)
            plt.legend()

            out_name = os.path.join(args.out_dir + args.out_name +
                                    '_' + name + '.png')
            plt.savefig(out_name, dpi=args.dpi, bbox_inches='tight')
            plt.close()

    else:
        # Plot unmasking or masking data (by binary or first probability map)
        fig, ax = plt.subplots()
        plt.xlabel(args.x_label)
        plt.ylabel(args.y_label)
        plt.title(args.title)

        ax.scatter(maps_data[0], maps_data[1], label=args.label,
                   color=args.colors[0], s=args.marker_size,
                   marker=args.marker, alpha=args.transparency)

        # Add data thresholded with the second probability map
        if args.in_prob_maps:
            ax.scatter(maps_prob[0], maps_prob[1], label=args.label_prob,
                       color=args.colors[1], s=args.marker_size,
                       marker=args.marker, alpha=args.transparency)

        plt.legend()

        if args.show_only:
            plt.show()
        else:
            plt.savefig(os.path.join(args.out_dir, args.out_name),
                        dpi=args.dpi, bbox_inches='tight')


if __name__ == "__main__":
    main()
