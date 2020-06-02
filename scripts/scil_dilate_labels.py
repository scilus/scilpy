#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dilate regions (with or without masking) from a labeled volume:
- "label_to_dilate" are regions that will dilate over
    "label_to_fill" if close enough to it ("distance").
- "label_to_dilate", by default (None) will be all
        non-"label_to_fill" and non-"label_not_to_dilate".
- "label_not_to_dilate" will not be changed, but will not dilate.
- "mask" is where the dilation is allowed (constrained)
    in addition to "background_label" (logical AND)

>>> scil_dilate_labels.py wmparc_t1.nii.gz wmparc_dil.nii.gz \\
    --label_to_fill 0 5001 5002 \\
    --label_not_to_dilate 4 43 10 11 12 49 50 51
"""

import argparse
import logging

import nibabel as nib
import numpy as np
from scipy.spatial.ckdtree import cKDTree

from scilpy.io.image import get_data_as_label, get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             assert_inputs_exist, assert_outputs_exist)

EPILOG = """
    References:
        [1] Al-Sharif N.B., St-Onge E., Vogel J.W., Theaud G.,
            Evans A.C. and Descoteaux M. OHBM 2019.
            Surface integration for connectome analysis in age prediction.
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_file',
                   help='Path of the volume (nii or nii.gz).')

    p.add_argument('out_file',
                   help='Output filename of the dilated labels.')

    p.add_argument('--distance', type=float, default=2.0,
                   help='Maximal distance to dilated (in mm) [%(default)s].')

    p.add_argument('--label_to_dilate', type=int, nargs='+', default=None,
                   help='Label list to dilate, by default it dilate all that\n'
                        ' are not in label_to_fill nor label_not_to_dilate.')

    p.add_argument('--label_to_fill', type=int, nargs='+', default=[0],
                   help='Background id / labels to be filled [%(default)s],\n'
                        ' the first one is given as output background value.')

    p.add_argument('--label_not_to_dilate', type=int, nargs='+', default=[],
                   help='Label list not to dilate.')

    p.add_argument('--mask',
                   help='Only dilate values inside the mask.')

    add_processes_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, args.in_file, optional=args.mask)
    assert_outputs_exist(parser, args, args.out_file)

    if args.nbr_processes is None:
        args.nbr_processes = -1

    # load volume
    volume_nib = nib.load(args.in_file)
    data = get_data_as_label(volume_nib)
    vox_size = np.reshape(volume_nib.header.get_zooms(), (1, 3))
    img_shape = data.shape

    # Check if in both: label_to_fill & not_to_fill
    fill_and_not = np.in1d(args.label_not_to_dilate, args.label_to_fill)
    if np.any(fill_and_not):
        logging.error("Error, both in not_to_dilate and to_fill: {}".format(
                      np.asarray(args.label_not_to_dilate)[fill_and_not]))

    # Create background mask
    is_background_mask = np.zeros(img_shape, dtype=np.bool)
    for i in args.label_to_fill:
        is_background_mask = np.logical_or(is_background_mask, data == i)

    # Create not_to_dilate mask (initialized to background)
    not_to_dilate = np.copy(is_background_mask)
    for i in args.label_not_to_dilate:

        not_to_dilate = np.logical_or(not_to_dilate, data == i)

    # Add mask
    if args.mask:
        mask_nib = nib.load(args.mask)
        mask_data = get_data_as_mask(mask_nib)
        to_dilate_mask = np.logical_and(is_background_mask, mask_data)
    else:
        to_dilate_mask = is_background_mask

    # Create label mask
    is_label_mask = ~not_to_dilate

    if args.label_to_dilate is not None:
        # Check if in both: to_dilate & not_to_dilate
        dil_and_not = np.in1d(args.label_to_dilate, args.label_not_to_dilate)
        if np.any(dil_and_not):
            logging.error("Error, both in dilate and Not to dilate: {}".format(
                          np.asarray(args.label_to_dilate)[dil_and_not]))

        # Check if in both: to_dilate & to_fill
        dil_and_fill = np.in1d(args.label_to_dilate, args.label_to_fill)
        if np.any(dil_and_fill):
            logging.error("Error, both in dilate and to fill: {}".format(
                          np.asarray(args.label_to_dilate)[dil_and_fill]))

        # Create new label to dilate list
        new_label_mask = np.zeros_like(data, dtype=np.bool)
        for i in args.label_to_dilate:
            new_label_mask = np.logical_or(new_label_mask, data == i)

        # Combine both new_label_mask and not_to_dilate
        is_label_mask = np.logical_and(new_label_mask, ~not_to_dilate)

    # Get the list of indices
    background_pos = np.argwhere(to_dilate_mask) * vox_size
    label_pos = np.argwhere(is_label_mask) * vox_size
    ckd_tree = cKDTree(label_pos)

    # Compute the nearest labels for each voxel of the background
    dist, indices = ckd_tree.query(
        background_pos, k=1, distance_upper_bound=args.distance,
        n_jobs=args.nbr_processes)

    # Associate indices to the nearest label (in distance)
    valid_nearest = np.squeeze(np.isfinite(dist))
    id_background = np.flatnonzero(to_dilate_mask)[valid_nearest]
    id_label = np.flatnonzero(is_label_mask)[indices[valid_nearest]]

    # Change values of those background
    data = data.flatten()
    data[id_background.T] = data[id_label.T]
    data = data.reshape(img_shape)

    # Save image
    nib.save(nib.Nifti1Image(data.astype(np.uint16), volume_nib.affine,
                             header=volume_nib.header),
             args.out_file)


if __name__ == "__main__":
    main()
