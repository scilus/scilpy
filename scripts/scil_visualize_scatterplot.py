#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to display scatter plot between two maps (ex. FA and MD).
This script can be also used for the correspondence between two maps
(ex. AD and RD). Two probability maps can be used to threshold maps.
Therefore, the --probability --prob_mask (WM) and --prob_mask_1 (GM) options
must be added.

For general scatter plot:
>>> scil_visualize_scatterplot.py FA.nii.gz MD.nii.gz out_filename_image.png

For tissue probability scatter plot:
>>> scil_visualize_scatterplot.py FA.nii.gz MD.nii.gz out_filename_image.png
    --probability --prob_mask probability_wm_map.nii.gz
    --prob_mask_2 probability_gm_map.nii.gz

To display specific label for probability scatter plot used:
    --label 'WM Threshold' --label_prob_2 'GM Threshold'

"""

import argparse
import copy

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import add_overwrite_arg, assert_inputs_exist


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_x_map',
                   help='Map in x axis, FA for example.')
    p.add_argument('in_y_map',
                   help='Map in y axis, MD for example.')
    p.add_argument('out_png',
                   help='Output filename for the figure.')
    p.add_argument('--mask',
                   help='Binary mask map. Use this mask to extract x and '
                        'y maps value from specific map or region: '
                        'wm_mask or roi_mask')
    p.add_argument('--out_dir',
                   help='')

    atlas = p.add_argument_group(title='Atlas options')
    atlas.add_argument('--in_atlas',
                       help='Path to the input atlas image.')
    atlas.add_argument('--in_atlas_lut',
                       help='Path of the LUT file corresponding to atlas '
                            'used to name the regions of interest.')
    atlas.add_argument('--specific_label', type=int, nargs='+', default=None,
                       help='Label list to use to do scatter plot. Label must'
                            ' corresponding tp atlas LUT file.')

    probmap = p.add_argument_group(title='Probability maps options')
    probmap.add_argument('--probability', action='store_true',
                         help='Compute and display specific scatter plot for '
                              'probability maps.')
    probmap.add_argument('--prob_mask',
                         help='Probability map, WM for example.')
    probmap.add_argument('--prob_mask_2',
                         help='Used to add a second probability map. '
                              'GM for example.')
    probmap.add_argument('--thr', default='0.9',
                         help='Use to apply threshold on probability mask.'
                              ' [%(default)s]')

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
    scat.add_argument('--label', default='Mask',
                      help='Use the provided info for the legend box '
                           'corresponding to mask or probability map. '
                           ' [%(default)s]')
    scat.add_argument('--label_prob_2', default=' ',
                      help='Use the provided info for the legend box '
                           'corresponding to the second probability map. '
                           ' [%(default)s]')
    scat.add_argument('--marker', default='.',
                      help='Use the provided info for the marker shape.'
                           ' [%(default)s]')
    scat.add_argument('--marker_size', default='15',
                      help='Use the provided info for the marker shape.'
                           ' [%(default)s]')
    scat.add_argument('--transparency', default='0.4',
                      help='Use the provided info for the marker shape.'
                           ' [%(default)s]')
    scat.add_argument('--dpi', default='300',
                      help='Use the provided info for the dpi resolution.'
                           ' [%(default)s]')
    scat.add_argument('--color_prob', nargs=2, metavar=('color1', 'color2'),
                      default=('r', 'b'))

    p.add_argument('--show_only', action='store_true',
                   help='Do not save the figure, only display.')

    add_overwrite_arg(p)

    return p


def load_data(images_list, mask=False, label=False):
    map = []
    for curr_map in images_list:
        load_image = nib.load(curr_map)

        if label:
            map.append(get_data_as_label(load_image))
            with open(args.in_atlas_lut) as f:
                label_dict = json.load(f)
            return map, zip(*label_dict.items())

        if mask:
            map.append(get_data_as_mask(load_image))
        else:
            map.append(load_image.get_fdata(dtype=np.float32))

    return map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_x_map, args.in_y_map)
    if args.probability:
        assert_inputs_exist(parser, args.prob_mask, args.prob_mask_2)

    if args.in_atlas:
        assert_inputs_exist(parser, args.in_atlas_lut)

    # Load images
    maps_image = [args.in_x_map, args.in_y_map]
    maps_data = load_data(maps_image)

    # Load and apply threshold from mask
    if args.mask:
        #mask_image = nib.load(args.mask)
        #mask_data = get_data_as_mask(mask_image)
        mask_data = load_data(args.mask, mask=True)
        for curr_map in maps_data:
            curr_map[np.where(mask_data == 0)] = np.nan

    if args.atlas:
        label_img_data, (lut_indices, lut_names) = load_data(args.in_atlas,
                                                                 label=True)
        if args.specific_label:
            for key in args.specific_label:
                label_indices, label_names = (lut_indices[key], lut_names[key])
        else:
            (label_indices, label_names) = (lut_indices, lut_names)

    if args.probability:
        if args.prob_mask_2 is None:
            tissue_image = args.prob_mask
        else:
            tissue_image = [args.prob_mask, args.prob_mask_2]
        prob_mask_data = load_data(tissue_image, mask=True)

        maps_data_prob = copy.deepcopy(maps_data)

        # Threshold probability images with tissue probability maps
        for curr_map in maps_data:
            curr_map[np.where(prob_mask_data[0] < args.thr)] = np.nan

        for curr_map in maps_data_prob:
            curr_map[np.where(prob_mask_data[1] < args.thr)] = np.nan

    # Scatter Plot
    fig, ax = plt.subplots()
    ax.scatter(maps_data[0], maps_data[1], label=args.label,
    color=args.color_prob[0], s=args.marker_size,
    marker=args.marker, alpha=args.transparency)

    if args.probability:
        ax.scatter(maps_data_prob[0], maps_data_prob[1],
        label=args.label_prob_2, color=args.color_prob[1],
        s=args.marker_size, marker=args.marker,
        alpha=args.transparency)
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.title(args.title)
    plt.legend()

    if args.in_atlas:
        for label, name in zip(label_indices, label_names):
            label = int(label)
            ax.scatter(maps_data[0], maps_data[1], label=name,
            color=args.color_prob[0], s=args.marker_size,
            marker=args.marker, alpha=args.transparency)

            plt.savefig(args.out_png+name, dpi=args.dpi, bbox_inches='tight')

    if args.show_only:
        plt.show()
    else:
        plt.savefig(args.out_png, dpi=args.dpi, bbox_inches='tight')


if __name__ == "__main__":
    main()
