#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Computes a very simple connectivity matrix, using the streamline count and the
position of the streamlines' endpoints.

This script is intented for exploration of your data. For a more thorough
computation (using the longest streamline segment), and for more options about
the weights of the matrix, see:
>> scil_connectivity_compute_matrices.py

Contrary to scil_connectivity_compute_matrices.py, works with an incomplete
parcellation (i.e. with streamlines ending in the background).

In the output figure, 4 matrices are shown, all using the streamline count:
    - Raw count
    - Raw count (log view)
    - Binary matrix (if at least 1 streamline connects the two regions)
    - Percentage of the total streamline count.

You may select which matrix to save to disk (as .npy) using options --binary or
--percentage. Default ouput matrix is the raw count.
"""

import argparse
import logging
import os.path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nibabel as nib
import numpy as np

from scilpy.connectivity.connectivity import \
    compute_triu_connectivity_from_labels
from scilpy.image.labels import get_data_as_labels

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, \
    add_verbose_arg, add_overwrite_arg, assert_headers_compatible, \
    add_reference_arg
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractogram',
                   help='Tractogram (trk or tck).')
    p.add_argument('in_labels',
                   help='Input nifti volume.')
    p.add_argument('out_matrix',
                   help="Out .npy file.")
    p.add_argument('out_labels',
                   help="Out .txt file. Will show the ordered labels (i.e. "
                        "the columns and lines' tags).")

    g = p.add_argument_group("Label management options")
    g.add_argument('--keep_background', action='store_true',
                   help="By default, the background (label 0) is not included "
                        "in the matrix. \nUse this option to keep it.")
    g.add_argument('--hide_labels', metavar='label', nargs='+',
                   help="Set given labels' weights to 0 in the matrix. \n"
                        "Their row and columns wil be kept but set to 0.")

    g = p.add_argument_group("Figure options")
    g.add_argument('--hide_fig', action='store_true',
                   help="If set, does not show the matrices with matplotlib "
                        "(you can still use --out_fig)")
    g.add_argument('--out_fig', metavar='file.png',
                   help="If set, saves the figure to file. \nExtension can be "
                        "any format understood by matplotlib (ex, .png).")

    g = p.add_argument_group("Output matrix (.npy) options")
    g = g.add_mutually_exclusive_group()
    g.add_argument('--binary', action='store_true',
                   help="If set, saves the result as binary. Else, the "
                        "streamline count is saved.")
    g.add_argument('--percentage', action='store_true')

    add_verbose_arg(p)
    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def prepare_figure_connectivity(matrix):
    matrix = np.copy(matrix)

    fig, axs = plt.subplots(2, 2)
    im = axs[0, 0].imshow(matrix)
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    axs[0, 0].set_title("Raw streamline count")

    im = axs[0, 1].imshow(matrix + np.min(matrix[matrix > 0]), norm=LogNorm())
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    axs[0, 1].set_title("Raw streamline count (log view)")

    matrix = matrix / matrix.sum() * 100
    im = axs[1, 0].imshow(matrix)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    axs[1, 0].set_title("Percentage of the total streamline count")

    matrix = matrix > 0
    axs[1, 1].imshow(matrix)
    axs[1, 1].set_title("Binary matrix: 1 if at least 1 streamline")

    plt.suptitle("Connectivity matrix: streamline count")


def main():
    p = _build_arg_parser()
    args = p.parse_args()
    logging.getLogger().setLevel(args.verbose)
    if args.verbose == 'DEBUG':
        # Currently, with debug, matplotlib prints a lot of stuff. Why??
        logging.getLogger().setLevel(logging.INFO)

    # Verifications
    tmp, ext = os.path.splitext(args.out_matrix)
    if ext != '.npy':
        p.error("out_matrix should have a .npy extension.")

    assert_inputs_exist(p, [args.in_labels, args.in_tractogram],
                        args.reference)
    assert_headers_compatible(p, [args.in_labels, args.in_tractogram], [],
                              args.reference)
    assert_outputs_exist(p, args, args.out_matrix, args.out_fig)

    # Loading
    in_sft = load_tractogram_with_reference(p, args, args.in_tractogram)
    in_img = nib.load(args.in_labels)
    data_labels = get_data_as_labels(in_img)

    # Computing
    matrix, ordered_labels, _, _ = \
        compute_triu_connectivity_from_labels(
            in_sft, data_labels, keep_background=args.keep_background,
            hide_labels=args.hide_labels)

    # Save figure will all versions of the matrix.
    if (not args.hide_fig) or args.out_fig is not None:
        prepare_figure_connectivity(matrix)

        if args.out_fig is not None:
            plt.savefig(args.out_fig)

    # Save matrix
    if args.binary:
        matrix = matrix > 0
    elif args.percentage:
        matrix = matrix / matrix.sum() * 100
    np.save(args.out_matrix, matrix)

    # Save labels
    with open(args.out_labels, "w") as text_file:
        for i, label in enumerate(ordered_labels):
            text_file.write("{} = {}\n".format(i, label))

    # Showing as last step. Everything else is done, so if user closes figure
    # it's fine.
    if not args.hide_fig:
        plt.show()


if __name__ == '__main__':
    main()
