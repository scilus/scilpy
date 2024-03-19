#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform tractogram(s) contained in the hdf5 output from a connectivity
script, using an affine/rigid transformation and nonlinear deformation
(optional).

See scil_tractogram_apply_transform.py to apply directly to a tractogram.

For more information on how to use the registration script, follow this link:
https://scilpy.readthedocs.io/en/latest/documentation/tractogram_registration.html

Formerly: scil_apply_transform_to_hdf5.py
"""

import argparse
import logging
import os

from dipy.io.stateful_tractogram import Space, Origin, StatefulTractogram
from dipy.io.utils import create_nifti_header, get_reference_info
import h5py
import nibabel as nib
import numpy as np

from scilpy.io.streamlines import reconstruct_streamlines_from_hdf5
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

    p.add_argument('--cut_invalid', action='store_true',
                   help='Cut invalid streamlines rather than removing them.\n'
                        'Keep the longest segment only.\n'
                        'By default, invalid streamline are removed.')

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
                _ = out_hdf5_file.create_group(key)

                # Copy group from in_hdf5, reconstruct the tractogram
                affine = in_hdf5_file.attrs['affine']
                dimensions = in_hdf5_file.attrs['dimensions']
                voxel_sizes = in_hdf5_file.attrs['voxel_sizes']
                streamlines = reconstruct_streamlines_from_hdf5(
                    in_hdf5_file, key)

                if len(streamlines) == 0:
                    continue
                header = create_nifti_header(affine, dimensions, voxel_sizes)
                moving_sft = StatefulTractogram(streamlines, header, Space.VOX,
                                                origin=Origin.TRACKVIS)

                # Load dps
                for dps_key in in_hdf5_file[key].keys():
                    if (dps_key not in ['data', 'offsets', 'lengths'] and
                            in_hdf5_file[key][dps_key].shape ==
                            in_hdf5_file[key]['offsets']):
                        moving_sft.data_per_streamline[dps_key] \
                            = in_hdf5_file[key][dps_key]

                # Main processing
                new_sft = transform_warp_sft(
                    moving_sft, transfo, target_img,
                    inverse=args.inverse,
                    deformation_data=deformation_data,
                    reverse_op=args.reverse_operation,
                    remove_invalid=not args.cut_invalid,
                    cut_invalid=args.cut_invalid)

                # Save result to the hdf5
                new_sft.to_vox()
                new_sft.to_corner()
                affine, dimensions, voxel_sizes, voxel_order = \
                    get_reference_info(target_img)
                out_hdf5_file.attrs['affine'] = affine
                out_hdf5_file.attrs['dimensions'] = dimensions
                out_hdf5_file.attrs['voxel_sizes'] = voxel_sizes
                out_hdf5_file.attrs['voxel_order'] = voxel_order

                # Get the data. Could use new_sft.streamlines._data. Avoiding
                # using hidden variable. Could find how to do the same with
                # _offsets.
                data = np.vstack(new_sft.streamlines).astype(np.float32)
                lengths = np.asarray([len(s) for s in new_sft.streamlines])

                group = out_hdf5_file[key]
                group.create_dataset('data', data=data)
                group.create_dataset('offsets',
                                     data=new_sft.streamlines._offsets)
                group.create_dataset('lengths', data=lengths)
                for dps_key in in_hdf5_file[key].keys():
                    if dps_key not in ['data', 'offsets', 'lengths']:
                        if in_hdf5_file[key][dps_key].shape \
                                == in_hdf5_file[key]['offsets']:
                            group.create_dataset(
                                dps_key,
                                data=new_sft.data_per_streamline[dps_key])
                        else:
                            group.create_dataset(
                                dps_key,
                                data=in_hdf5_file[key][dps_key])


if __name__ == "__main__":
    main()
