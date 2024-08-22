#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO
"""

import argparse
import json
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             add_json_args, assert_outputs_exist,
                             add_verbose_arg, add_reference_arg,
                             assert_headers_compatible)
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.metrics_tools import compute_lesion_stats


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_image',
                   help='Lesions file as mask OR labels (.nii.gz).')
    p.add_argument('out_image',
                   help='TODO')

    p.add_argument('--nb_ring', type=int,
                   help='Integer representing the number of rings to be '
                        'created.')
    p.add_argument('--ring_thickness', type=int,
                   help='Integer representing the thickness of the rings to be '
                        'created. Used for voxel dilation passes.')
    # TODO split 4D into many files

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))


    assert_inputs_exist(parser, args.in_image)
    assert_outputs_exist(parser, args, args.out_image)

    lesion_img = nib.load(args.in_image)
    lesion_atlas = get_data_as_labels(lesion_img)

    if np.unique(lesion_atlas).size == 1:
        raise ValueError('Input lesion map is empty.')
    is_binary = True if np.unique(lesion_atlas).size == 2 else False
    print(is_binary)
    labels = np.unique(lesion_atlas)[1:]
    nawm = np.zeros(lesion_atlas.shape + (len(labels),), dtype=np.uint16)
    for i, label in enumerate(labels):
        nawm[..., i] = i + 1

    if is_binary:
        nawm = np.squeeze(nawm)

    nib.save(nib.Nifti1Image(nawm, lesion_img.affine), args.out_image)



if __name__ == "__main__":
    main()
