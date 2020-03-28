#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Warp tractogram using a non linear deformation from an ANTs deformation map.

For more information on how to use the various registration scripts
see the doc/tractogram_registration.md readme file

Applying transformation to tractogram can lead to invalid streamlines (out of
the bbox), three strategies are available:
- default, crash at saving if invalid streamlines are present
- --keep_invalid, save invalid streamlines. Leave it to the user to run
    scil_remove_invalid_streamlines.py if needed.
- --remove_invalid, automatically remove invalid streamlines before saving.
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
from scilpy.utils.streamlines import warp_streamlines


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('moving_tractogram',
                   help='Path of the tractogram to be transformed.')
    p.add_argument('target_file',
                   help='Path of the reference file (trk or nii).')
    p.add_argument('deformation',
                   help='Path of the file containing deformation field.')
    p.add_argument('out_tractogram',
                   help='Output filename of the transformed tractogram.')

    invalid = p.add_mutually_exclusive_group()
    invalid.add_argument('--remove_invalid', action='store_true',
                         help='Remove the streamlines landing out of the bbox.')
    invalid.add_argument('--keep_invalid', action='store_true',
                         help='Keep the streamlines landing out of the bbox.')

    add_overwrite_arg(p)
    add_reference_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.moving_tractogram, args.target_file,
                                 args.deformation])
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.moving_tractogram,
                                         bbox_check=False)

    deformation = nib.load(args.deformation)
    deformation_data = np.squeeze(deformation.get_fdata())

    if not is_header_compatible(sft, deformation):
        parser.error('Input tractogram/reference do not have the same spatial '
                     'attribute as the deformation field.')

    # Warning: Apply warp in-place
    moved_streamlines = warp_streamlines(sft, deformation_data)
    new_sft = StatefulTractogram(moved_streamlines, args.target_file,
                                 Space.RASMM,
                                 data_per_point=sft.data_per_point,
                                 data_per_streamline=sft.data_per_streamline)

    if args.remove_invalid:
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
