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

>>> scil_labels_dilate.py wmparc_t1.nii.gz wmparc_dil.nii.gz \\
    --label_to_fill 0 5001 5002 \\
    --label_not_to_dilate 4 43 10 11 12 49 50 51

Formerly: scil_dilate_labels.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels, dilate_labels
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             assert_inputs_exist, add_verbose_arg,
                             assert_outputs_exist, assert_headers_compatible)

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
                   help='Maximal distance to dilate (in mm) [%(default)s].')
    p.add_argument('--labels_to_dilate', type=int, nargs='+', default=None,
                   help='Label list to dilate. By default it dilates all \n'
                        'labels not in labels_to_fill nor in '
                        'labels_not_to_dilate.')
    p.add_argument('--labels_to_fill', type=int, nargs='+', default=[0],
                   help='Background id / labels to be filled [%(default)s],\n'
                        ' the first one is given as output background value.')
    p.add_argument('--labels_not_to_dilate', type=int, nargs='+', default=[],
                   help='Label list not to dilate.')
    p.add_argument('--mask',
                   help='Only dilate values inside the mask.')

    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_file, optional=args.mask)
    assert_outputs_exist(parser, args, args.out_file)
    assert_headers_compatible(parser, args.in_file, optional=args.mask)

    if args.nbr_processes is None:
        args.nbr_processes = -1

    # load volume
    volume_nib = nib.load(args.in_file)
    data = get_data_as_labels(volume_nib)
    vox_size = np.reshape(volume_nib.header.get_zooms(), (1, 3))

    mask_data = get_data_as_mask(nib.load(args.mask)) if args.mask else None

    data = dilate_labels(data, vox_size, args.distance, args.nbr_processes,
                         labels_to_dilate=args.labels_to_dilate,
                         labels_not_to_dilate=args.labels_not_to_dilate,
                         labels_to_fill=args.labels_to_fill,
                         mask=mask_data)

    # Save image
    nib.save(nib.Nifti1Image(data.astype(np.uint16), volume_nib.affine,
                             header=volume_nib.header),
             args.out_file)


if __name__ == "__main__":
    main()
