#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transform tractogram using an affine/rigid transformation.

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
from dipy.tracking.streamline import transform_streamlines
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist, assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('moving_tractogram',
                   help='Path of the tractogram to be transformed.')
    p.add_argument('target_file',
                   help='Path of the reference target file (trk or nii).')
    p.add_argument('transformation',
                   help='Path of the file containing the 4x4 \n'
                        'transformation, matrix (*.txt).'
                        'See the script description for more information.')
    p.add_argument('out_tractogram',
                   help='Output tractogram filename (transformed data).')

    p.add_argument('--inverse', action='store_true',
                   help='Apply the inverse transformation.')

    invalid = p.add_mutually_exclusive_group()
    invalid.add_argument('--remove_invalid', action='store_true',
                         help='Remove the streamlines landing out of the bbox.')
    invalid.add_argument('--keep_invalid', action='store_true',
                         help='Keep the streamlines landing out of the bbox.')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.moving_tractogram, args.target_file,
                                 args.transformation])
    assert_outputs_exist(parser, args, args.out_tractogram)

    moving_sft = load_tractogram_with_reference(parser, args,
                                                args.moving_tractogram,
                                                bbox_check=False)

    transfo = np.loadtxt(args.transformation)
    if args.inverse:
        transfo = np.linalg.inv(transfo)

    moved_streamlines = transform_streamlines(moving_sft.streamlines,
                                              transfo)
    new_sft = StatefulTractogram(moved_streamlines, args.target_file,
                                 Space.RASMM,
                                 data_per_point=moving_sft.data_per_point,
                                 data_per_streamline=moving_sft.data_per_streamline)

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
