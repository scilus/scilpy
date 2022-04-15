#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assign a color to each streamline point based on its p-value. The p-values
are assumed to be in the same order as the streamline points. If a streamline
has less points than the number of p-values, the streamline is resampled
to have the same number of points. The number of points can also be set
by the user with --resample.

The p-values input format is a whitespace-separated list of values saved
in a .txt file.

The resulting tractogram can be visualized with MI-Brain. A PNG of the colormap
is also saved.
"""
import argparse
import nibabel as nib
import numpy as np
import os

from dipy.tracking.streamline import set_number_of_points
from fury import colormap

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


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
                   help='Output colormap image (.png).')

    p.add_argument('--colormap', default='plasma',
                   help='Colormap to use for coloring the streamlines.')
    p.add_argument('--resample', type=int,
                   help='Optionally resample the streamlines.')

    add_overwrite_arg(p)
    return p


def resample_streamlines_from_pvalues(streamlines, pvalues,
                                      override_resample=None):
    """
    Resample each streamline so that it has at least the same
    number of points as the p-values vector. If override_resample
    is set, the number of points is set to this value.

    Parameters
    ----------
    streamlines : list of ndarray
        List of streamlines.
    pvalues : ndarray
        Array of p-values.
    override_resample : int, optional
        If set, the number of points is set to this value.

    Returns
    -------
    resampled_streamlines: list of ndarray
        List of resampled streamlines.
    mapped_pvalues: ndarray
        Array of p-values mapped to the resampled streamlines.
    """
    n_timepoints = len(pvalues)
    f_interp = interp1d(np.linspace(0.0, 1.0, n_timepoints), pvalues)

    mapped_pvals = []
    resampled_strl = []
    for strl in streamlines:
        if len(strl) < n_timepoints or override_resample is not None:
            new_n_points = override_resample if override_resample is not None\
                 else n_timepoints
            strl = set_number_of_points(strl, new_n_points)

        # compute interpolation timepoints
        distances = np.sqrt(np.sum((strl[1:] - strl[:-1])**2, axis=-1))
        distances = np.append([0], distances)
        weights = np.cumsum(distances)
        weights /= weights[-1]
        pvals = f_interp(weights)
        mapped_pvals.append(pvals)
        resampled_strl.append(strl)

    return resampled_strl, mapped_pvals


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bundle, args.in_pvalues])
    assert_outputs_exist(parser, args,
                         [args.out_tractogram, args.out_colormap])

    # Assert that the output tractogram is in .trk format
    _, ext = os.path.splitext(args.out_tractogram)
    if ext != '.trk':
        parser.error("Output tractogram file must have .trk extension.")

    pvalues = np.loadtxt(args.in_pvalues).reshape((-1,))
    tractogram = nib.streamlines.load(args.in_bundle)

    # resample streamlines if necessary
    streamlines, pvals =\
        resample_streamlines_from_pvalues(tractogram.streamlines,
                                          pvalues,
                                          args.resample)

    # assign colors to streamlines
    color = colormap.create_colormap(np.ravel(pvals),
                                     args.colormap, auto=False)
    sft = StatefulTractogram(streamlines, tractogram, Space.RASMM)
    sft.data_per_point['color'] = sft.streamlines
    sft.data_per_point['color']._data = color * 255
    save_tractogram(sft, args.out_tractogram)

    # output colormap to png file
    n_steps_cmap = 256
    xticks = np.linspace(0, 1, n_steps_cmap)
    gradient = colormap.create_colormap(xticks,
                                        args.colormap,
                                        auto=False)
    gradient = gradient[None, ...]  # 2D RGB image
    _, ax = plt.subplots(1, 1, figsize=(10, 1), dpi=100)
    ax.imshow(gradient, aspect=10)
    ax.set_xticks([0, n_steps_cmap - 1])
    ax.set_xticklabels(['0.0', '1.0'])
    ax.set_yticks([])
    plt.savefig(args.out_colormap)


if __name__ == '__main__':
    main()
