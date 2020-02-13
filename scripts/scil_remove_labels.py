#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)

DESCRIPTION = """
    Script to remove specific labels from a atlas volumes.

    >>> scil_remove_labels.py DKT_labels.nii out_labels.nii.gz  -i  44 53
    """

EPILOG = """
    References:
        [1] Al-Sharif N.B., St-Onge E., Vogel J.W., Theaud G.,
            Evans A.C. and Descoteaux M. OHBM 2019.
            Surface integration for connectome analysis in age prediction.
    """


def _build_args_parser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Input labels volume.')

    p.add_argument('output',
                   help='Output labels volume.')

    p.add_argument('-i', '--indices', type=int, nargs='+', required=True,
                   help='List of labels indices to remove.')

    p.add_argument('--background', type=int, default=0,
                   help='Background id, excluded from output [%(default)s],\n'
                        ' the value is used as output background value.')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.input)
    assert_outputs_exist(parser, args, args.output)

    # Load volume
    volume_img = nib.load(args.input)
    labels_volume = volume_img.get_data()

    # Remove given labels from the volume
    for index in np.unique(args.indices):
        mask = labels_volume == index
        labels_volume[mask] = args.background
        if np.count_nonzero(mask) == 0:
            logging.warning("Label {} was not in the volume".format(index))

    # Save final volume
    nii = nib.Nifti1Image(labels_volume, volume_img.affine, volume_img.header)
    nib.save(nii, args.output)


if __name__ == "__main__":
    main()
