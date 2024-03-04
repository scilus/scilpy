#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to remove specific labels from an atlas volume.

    >>> scil_labels_remove.py DKT_labels.nii out_labels.nii.gz -i 5001 5002

Formerly: scil_remove_labels.py
"""


import argparse
import logging

import nibabel as nib

from scilpy.image.labels import get_data_as_labels, remove_labels
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             add_verbose_arg, assert_outputs_exist)
EPILOG = """
    References:
        [1] Al-Sharif N.B., St-Onge E., Vogel J.W., Theaud G.,
            Evans A.C. and Descoteaux M. OHBM 2019.
            Surface integration for connectome analysis in age prediction.
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_labels',
                   help='Input labels volume.')
    p.add_argument('out_labels',
                   help='Output labels volume.')

    p.add_argument('-i', '--indices', type=int, nargs='+', required=True,
                   help='List of labels indices to remove.')
    p.add_argument('--background', type=int, default=0,
                   help='Integer used for removed labels [%(default)s].')
    
    add_verbose_arg(p)
    add_overwrite_arg(p)
    
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_labels)
    assert_outputs_exist(parser, args, args.out_labels)

    # Load volume
    label_img = nib.load(args.in_labels)
    labels_volume = get_data_as_labels(label_img)

    labels_volume = remove_labels(labels_volume, args.indices, args.background)

    # Save final volume
    nii = nib.Nifti1Image(labels_volume, label_img.affine, label_img.header)
    nib.save(nii, args.out_labels)


if __name__ == "__main__":
    main()
