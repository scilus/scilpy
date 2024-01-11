#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform tractogram using an affine/rigid transformation and nonlinear
deformation (optional).

For more information on how to use the registration script, follow this link:
https://scilpy.readthedocs.io/en/latest/documentation/tractogram_registration.html

Applying transformation to tractogram can lead to invalid streamlines (out of
the bounding box), three strategies are available:
1) default, crash at saving if invalid streamlines are present
2) --keep_invalid, save invalid streamlines. Leave it to the user to run
    scil_tractogram_remove_invalid.py if needed.
3) --remove_invalid, automatically remove invalid streamlines before saving.
    Should not remove more than a few streamlines.
4) --cut_invalid, automatically cut invalid streamlines before saving.

Example:
To apply transform from ANTS to tractogram. If the ANTS commands was
MOVING->REFERENCE, this will bring a tractogram from MOVING->REFERENCE
scil_tractogram_apply_transform.py ${MOVING_FILE} ${REFERENCE_FILE}
                                   0GenericAffine.mat ${OUTPUT_NAME}
                                   --inverse
                                   --in_deformation 1InverseWarp.nii.gz

If the ANTS commands was MOVING->REFERENCE, this will bring a tractogram
from REFERENCE->MOVING
scil_tractogram_apply_transform.py ${MOVING_FILE} ${REFERENCE_FILE}
                                   0GenericAffine.mat ${OUTPUT_NAME}
                                   --in_deformation 1Warp.nii.gz
                                   --reverse_operation

Formerly: scil_apply_transform_to_tractogram.py
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram
import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)
from scilpy.tractograms.tractogram_operations import transform_warp_sft


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_moving_tractogram',
                   help='Path of the tractogram to be transformed.\n'
                        'Bounding box validity will not be checked (could '
                        'contain invalid streamlines).')
    p.add_argument('in_target_file',
                   help='Path of the reference target file (trk or nii).')
    p.add_argument('in_transfo',
                   help='Path of the file containing the 4x4 \n'
                        'transformation, matrix (.txt, .npy or .mat).')
    p.add_argument('out_tractogram',
                   help='Output tractogram filename (transformed data).')

    p.add_argument('--inverse', action='store_true',
                   help='Apply the inverse linear transformation.')
    p.add_argument('--in_deformation',
                   help='Path to the file containing a deformation field.')
    p.add_argument('--reverse_operation', action='store_true',
                   help='Apply the transformation in reverse (see doc),'
                        'warp first, then linear.')
    invalid = p.add_mutually_exclusive_group()
    invalid.add_argument('--cut_invalid', action='store_true',
                         help='Cut invalid streamlines rather than removing '
                              'them.\nKeep the longest segment only.')
    invalid.add_argument('--remove_invalid', action='store_true',
                         help='Remove the streamlines landing out of the '
                              'bounding box.')
    invalid.add_argument('--keep_invalid', action='store_true',
                         help='Keep the streamlines landing out of the '
                              'bounding box.')

    p.add_argument('--no_empty', action='store_true',
                   help='Do not write file if there is no streamline.\n'
                        'You may save an empty file if you use '
                        'remove_invalid.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    assert_inputs_exist(parser, [args.in_moving_tractogram,
                                 args.in_target_file,
                                 args.in_transfo], args.in_deformation)
    assert_outputs_exist(parser, args, args.out_tractogram)

    args.bbox_check = False  # Adding manually bbox_check argument.
    moving_sft = load_tractogram_with_reference(parser, args,
                                                args.in_moving_tractogram)

    transfo = load_matrix_in_any_format(args.in_transfo)
    deformation_data = None
    if args.in_deformation is not None:
        deformation_data = np.squeeze(nib.load(
            args.in_deformation).get_fdata(dtype=np.float32))

    new_sft = transform_warp_sft(moving_sft, transfo,
                                 args.in_target_file,
                                 inverse=args.inverse,
                                 reverse_op=args.reverse_operation,
                                 deformation_data=deformation_data,
                                 remove_invalid=args.remove_invalid,
                                 cut_invalid=args.cut_invalid)

    if len(new_sft.streamlines) == 0:
        if args.no_empty:
            logging.debug("The file {} won't be written "
                          "(0 streamline).".format(args.out_tractogram))
            return

    if args.keep_invalid:
        if not new_sft.is_bbox_in_vox_valid():
            logging.warning('Saving tractogram with invalid streamlines.')
        save_tractogram(new_sft, args.out_tractogram, bbox_valid_check=False)
    else:
        if not new_sft.is_bbox_in_vox_valid():
            logging.warning('Removing invalid streamlines before '
                            'saving tractogram.')
            new_sft.remove_invalid_streamlines()
        save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
