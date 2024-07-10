#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Script to obtain labels from a binary mask which contains multiple blobs.
"""


import argparse
import logging

import nibabel as nib

from scipy import ndimage as ndi

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             add_verbose_arg, assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_mask', type=str, help='Input mask file.')
    p.add_argument('out_labels', type=str, help='Output label file.')

    p.add_argument('--labels', nargs='+', default=[], type=int,
                   help='Labels to assign to each blobs in the mask. '
                        'Excludes the background label.')
    p.add_argument('--background_label', default=0, type=int,
                   help='Label for the background. [%(default)s]')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_mask)
    assert_outputs_exist(parser, args, args.out_labels)

    mask_img = nib.load(args.in_mask)
    mask_data = get_data_as_mask(mask_img)

    structures, nb_structures = ndi.label(mask_data)

    if args.labels:
        if len(args.labels) != nb_structures:
            parser.error("Number of labels ({}) does not match the number of "
                         "blobs in the mask ({}).".format(len(args.labels),
                                                          nb_structures))
        for idx, label in enumerate(args.labels):
            structures[structures == idx + 1] = label
    if args.background_label:
        structures[structures == 0] = args.background_label

    out_img = nib.Nifti1Image(structures.astype(float), mask_img.affine)
    nib.save(out_img, args.out_labels)


if __name__ == "__main__":
    main()
