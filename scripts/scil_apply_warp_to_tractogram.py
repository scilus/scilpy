#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Warp tractogram using a non linear deformation from an ANTs deformation field.

For more information on how to use the various registration scripts
see the doc/tractogram_registration.md readme file

Applying a deformation field to tractogram can lead to invalid streamlines (out
of the bounding box), three strategies are available:
1) default, crash at saving if invalid streamlines are present
2) --keep_invalid, save invalid streamlines. Leave it to the user to run
    scil_remove_invalid_streamlines.py if needed.
3) --remove_invalid, automatically remove invalid streamlines before saving.
    Should not remove more than a few streamlines.
"""

import argparse
import logging

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.utils.streamlines import warp_streamlines, cut_invalid_streamlines


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_moving_tractogram',
                   help='Path to the tractogram to be transformed.')
    p.add_argument('in_target_file',
                   help='Path to the reference target file (trk or nii).')
    p.add_argument('in_deformation',
                   help='Path to the file containing a deformation field.')
    p.add_argument('out_tractogram',
                   help='Output filename of the transformed tractogram.')

    invalid = p.add_mutually_exclusive_group()
    invalid.add_argument('--cut_invalid', action='store_true',
                         help='Cut invalid streamlines rather than removing them.\n'
                         'Keep the longest segment only.')
    invalid.add_argument('--remove_invalid', action='store_true',
                         help='Remove the streamlines landing out of the '
                              'bounding box.')
    invalid.add_argument('--keep_invalid', action='store_true',
                         help='Keep the streamlines landing out of the '
                              'bounding box.')

    add_overwrite_arg(p)
    add_reference_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_moving_tractogram, args.in_target_file,
                                 args.in_deformation])
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_moving_tractogram,
                                         bbox_check=False)

    deformation = nib.load(args.in_deformation)
    deformation_data = np.squeeze(deformation.get_fdata())

    if not is_header_compatible(sft, deformation):
        parser.error('Input tractogram/reference do not have the same spatial '
                     'attribute as the deformation field.')

    # Warning: Apply warp in-place
    moved_streamlines = warp_streamlines(sft, deformation_data)
    new_sft = StatefulTractogram(moved_streamlines, args.in_target_file,
                                 Space.RASMM,
                                 data_per_point=sft.data_per_point,
                                 data_per_streamline=sft.data_per_streamline)

    if args.cut_invalid:
        new_sft = cut_invalid_streamlines(new_sft)
        save_tractogram(new_sft, args.out_tractogram)
    elif args.remove_invalid:
        ori_len = len(new_sft)
        new_sft.remove_invalid_streamlines()
        logging.warning('Removed {} invalid streamlines.'.format(
            ori_len - len(new_sft)))
        save_tractogram(new_sft, args.out_tractogram)
    elif args.keep_invalid:
        if not new_sft.is_bbox_in_vox_valid():
            logging.warning('Saving tractogram with invalid streamlines.')
        save_tractogram(new_sft, args.out_tractogram, bbox_valid_check=False)
    else:
        save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
