# -*- coding: utf-8 -*-
"""
Script to display scatter plot between two maps (ex. FA and MD).
This script can be also used for the correspondence between two Myelin maps
(ex. ihMTR and MTsat). For the myelin, the WM and GM probability maps are
used to threshold the myelin maps. Therefore, the --myelin --in_wm and --in_gm
options must be added.

For general scatter plot:
>>> scil_visualize_scatterplot.py ihMT_map.nii.gz
MT_map.nii.gz probability_wm_map.nii.gz probability_gm_map.nii.gz
out_filename_image.png

For Myelin scatter plot:
>>> scil_visualize_scatterplot.py ihMTR_map.nii.gz MTsat_map.nii.gz
    out_filename_image.png --myelin --in_wm probability_wm_map.nii.gz
    --in_gm probability_gm_map.nii.gz

To display specific label for myelin scatter plot used:
    --label 'WM Threshold' --label_myelin 'GM Threshold'

"""

import argparse
import copy

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_x_map',
                   help='Map in x axis, FA or ihMTR for exemple.')
    p.add_argument('in_y_map',
                   help='Map in y axis, MD or MTsat for exemple.')
    p.add_argument('out_png',
                   help='Output filename for the figure.')
    p.add_argument('--in_mask',
                   help='Binary mask map. Use this mask to extract x and '
                        'y maps value from specific map or region: '
                        'mask_wm or roi_mask')

    myelo = p.add_argument_group(title='Myelin options')
    myelo.add_argument('--myelin', action='store_true',
                       help='Compute and display specific scatter plot for '
                            'myelin maps.')
    myelo.add_argument('--in_wm',
                       help='Probability WM map.')
    myelo.add_argument('--in_gm',
                       help='Probability GM map.')

    scat = p.add_argument_group(title='Scatter plot options')
    scat.add_argument('--title',
                      default='Scatter Plot',
                      help='Use the provided info to title name. '
                           ' [%(default)s]')
    scat.add_argument('--x_label', default='x',
                      help='Use the provided info to name x axis. '
                           ' [%(default)s]')
    scat.add_argument('--y_label', default='y',
                      help='Use the provided info to name y axis. '
                           ' [%(default)s]')
    scat.add_argument('--label', default=' ',
                      help='Use the provided info to legend. '
                           ' [%(default)s]')
    scat.add_argument('--label_myelin', default='GM Threshold',
                      help='Use the provided info to legend myelin map. '
                           ' Coudl be add for Myeline scatter plot. '
                           '[%(default)s]')
    scat.add_argument('--marker', default='.',
                      help='Use the provided info to marker shape.'
                           ' [%(default)s]')

    p.add_argument('--show_only', action='store_true',
                   help='Do not save the figure, only display.')

    add_overwrite_arg(p)

    return p


def load_maps(images_list):
    map = []
    for curr_map in images_list:
        load_image = nib.load(curr_map)
        map.append(load_image.get_fdata(dtype=np.float32))
    return map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_x_map, args.in_y_map)
    if args.myelin:
        assert_inputs_exist(parser, args.in_gm, args.in_wm)

    # Load images
    maps_image = [args.in_x_map, args.in_y_map]
    maps_data = load_maps(maps_image)

    if args.in_mask:
        # Load and apply threshold from mask
        mask_image = nib.load(args.in_mask)
        mask_data = get_data_as_mask(mask_image)
        for curr_map in maps_data:
            curr_map[np.where(mask_data == 0)] = np.nan

    if args.myelin:
        tissue_image = [args.in_wm, args.in_gm]
        tissue_data = load_maps(tissue_image)
        maps_data_gm_thr = copy.deepcopy(maps_data)

        # Threshold myelin images with tissue probability maps
        # White Matter threshold
        for curr_map in maps_data:
            curr_map[np.where(tissue_data[0] < 0.9)] = np.nan
        # Grey Matter threshold
        for curr_map in maps_data_gm_thr:
            curr_map[np.where(tissue_data[1] < 0.9)] = np.nan

    # Scatter Plot
    fig, ax = plt.subplots()
    ax.scatter(maps_data[0], maps_data[1], label=args.label, color='b',
               s=15, marker=args.marker, alpha=0.4)
    if args.myelin:
        ax.scatter(maps_data_gm_thr[0], maps_data_gm_thr[1],
                   label=args.label_myelin, color='r', s=15,
                   marker=args.marker, alpha=0.4)
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.title(args.title)
    plt.legend()

    if args.show_only:
        plt.show()
    else:
        plt.savefig(args.out_png, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
