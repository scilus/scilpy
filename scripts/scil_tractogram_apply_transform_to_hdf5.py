#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform tractogram(s) contained in the hdf5 output from a connectivity
script, using an affine/rigid transformation and nonlinear deformation
(optional).

See scil_tractogram_apply_transform.py to apply directly to a tractogram.

For more information on how to use the registration script, follow this link:
https://scilpy.readthedocs.io/en/latest/documentation/tractogram_registration.html

Or use >> scil_tractogram_apply_transform.py --help

Formerly: scil_apply_transform_to_hdf5.py
"""

import argparse
import logging
import os

import h5py
import nibabel as nib
import numpy as np

from scilpy.io.hdf5 import (reconstruct_sft_from_hdf5,
                            construct_hdf5_from_sft)
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

    p.add_argument('in_hdf5',
                   help='Path of the hdf5 containing the moving tractogram, '
                        'to be transformed. (.h5 extension).')
    p.add_argument('in_target_file',
                   help='Path of the reference target file (.trk or .nii).')
    p.add_argument('in_transfo',
                   help='Path of the file containing the 4x4 \n'
                        'transformation, matrix (.txt, .npy or .mat).')
    p.add_argument('out_hdf5',
                   help='Output tractogram filename (transformed data).')

    g = p.add_argument_group("Transformation options")
    g.add_argument('--inverse', action='store_true',
                   help='Apply the inverse linear transformation.')
    g.add_argument('--in_deformation', metavar='file',
                   help='Path to the file containing a deformation field.')
    g.add_argument('--reverse_operation', action='store_true',
                   help='Apply the transformation in reverse (see doc), warp\n'
                        'first, then linear.')

    g = p.add_argument_group("Management of invalid streamlines")
    invalid = g.add_mutually_exclusive_group()
    invalid.add_argument('--cut_invalid', action='store_true',
                         help='Cut invalid streamlines rather than removing '
                              'them.\nKeep the longest segment only.')
    invalid.add_argument('--remove_invalid', action='store_true',
                         help='Remove the streamlines landing out of the '
                              'bounding box.')
    invalid.add_argument('--keep_invalid', action='store_true',
                         help='Keep the streamlines landing out of the '
                              'bounding box.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, [args.in_hdf5, args.in_target_file,
                                 args.in_transfo],
                        [args.in_deformation, args.reference])
    assert_outputs_exist(parser, args, args.out_hdf5)

    # HDF5 will not overwrite the file
    if os.path.isfile(args.out_hdf5):
        os.remove(args.out_hdf5)

    # Loading
    transfo = load_matrix_in_any_format(args.in_transfo)
    deformation_data = None
    if args.in_deformation is not None:
        deformation_data = np.squeeze(nib.load(
            args.in_deformation).get_fdata(dtype=float))

    # Processing
    with h5py.File(args.in_hdf5, 'r') as in_hdf5_file:
        with h5py.File(args.out_hdf5, 'a') as out_hdf5_file:
            target_img = nib.load(args.in_target_file)

            # For each bundle / tractogram in the hdf5:
            for key in in_hdf5_file.keys():
                # Get the bundle as sft
                moving_sft, _ = reconstruct_sft_from_hdf5(
                    in_hdf5_file, key, load_dps=True, load_dpp=False)
                if moving_sft is None:
                    continue

                # Main processing
                new_sft = transform_warp_sft(
                    moving_sft, transfo, target_img,
                    inverse=args.inverse,
                    deformation_data=deformation_data,
                    reverse_op=args.reverse_operation,
                    remove_invalid=args.remove_invalid,
                    cut_invalid=args.cut_invalid)

                # Default is to crash if invalid.
                if args.keep_invalid:
                    if not new_sft.is_bbox_in_vox_valid():
                        logging.warning(
                            'Saving tractogram with invalid streamlines.')
                else:
                    # Here, there should be no invalid streamlines left. Either
                    # option = to crash, or remove/cut, already managed.
                    if not new_sft.is_bbox_in_vox_valid():
                        raise ValueError(
                            "The result has invalid streamlines. Please "
                            "chose --keep_invalid, --cut_invalid or "
                            "--remove_invalid.")

                # Save result to the hdf5
                construct_hdf5_from_sft(out_hdf5_file, new_sft, key,
                                        save_dps=True, save_dpp=False)


if __name__ == "__main__":
    main()
