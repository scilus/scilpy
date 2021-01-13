#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to display scatter plot between two maps (ex. FA and MD).
This script can be also used for the correspondence between two maps
(ex. AD and RD). Two probability maps can be used to threshold the myelin maps.
Therefore, the --probability --prob_mask_1 (WM) and --prob_mask_1 (GM) options
must be added.

For general scatter plot:
>>> scil_visualize_scatterplot.py FA.nii.gz MD.nii.gz out_filename_image.png

For tissue probability scatter plot:
>>> scil_visualize_scatterplot.py FA.nii.gz MD.nii.gz out_filename_image.png
    --probability --prob_mask_1 probability_wm_map.nii.gz
    --prob_mask_2 probability_gm_map.nii.gz

To display specific label for myelin scatter plot used:
    --label 'WM Threshold' --label_myelin 'GM Threshold'

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

    g2.add_argument('--axis_text_size', nargs=2, metavar=('X_SIZE', 'Y_SIZE'),
                    default=(10, 10))

    probmap = p.add_argument_group(title='Probability maps options')
    probmap.add_argument('--probability', action='store_true',
                       help='Compute and display specific scatter plot for '
                            'myelin maps.')
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
    scat.add_argument('--label', default=' ',
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


def load_data(images_list, prob_mask=False):
    map = []
    for curr_map in images_list:
        load_image = nib.load(curr_map)
        if prob_mask:
            map.append(get_data_as_mask(mask_image))
        else:
            map.append(load_image.get_fdata(dtype=np.float32))
    return map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_x_map, args.in_y_map)
    if args.probability:
        assert_inputs_exist(parser, args.prob_mask, args.prob_mask_2)

    # Load images
    maps_image = [args.in_x_map, args.in_y_map]
    maps_data = load_data(maps_image)

    if args.mask:
        # Load and apply threshold from mask
        mask_image = nib.load(args.mask)
        mask_data = get_data_as_mask(mask_image)
        for curr_map in maps_data:
            curr_map[np.where(mask_data == 0)] = np.nan

    if args.probability:
        # Load images
        if args.prob_mask_2 is None:
            tissue_image = args.prob_mask
        else:
            tissue_image = [args.prob_mask, args.prob_mask_2]
        prob_mask_data = load_data(tissue_image, prob_mask=True)

        # Copy to apply two different probability maps in the same data
        maps_data_prob = copy.deepcopy(maps_data)

        # Threshold myelin images with tissue probability maps
        # White Matter threshold
        for curr_map in maps_data:
            curr_map[np.where(prob_mask_data[0] < args.thr)] = np.nan
        # Grey Matter threshold
        for curr_map in map_data_prob:
            curr_map[np.where(prob_mask_data[1] < args.thr)] = np.nan

    # Scatter Plot
    fig, ax = plt.subplots()
    ax.scatter(maps_data[0], maps_data[1], label=args.label_prob_1,
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

    if args.show_only:
        plt.show()
    else:
        plt.savefig(args.out_png, dpi=args.dpi, bbox_inches='tight')


if __name__ == "__main__":
    main()
