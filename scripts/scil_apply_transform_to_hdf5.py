#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform connectivity hdf5 (.h5) using an affine/rigid transformation and
nonlinear deformation (optional).

For more information on how to use the registration script, follow this link:
https://scilpy.readthedocs.io/en/latest/documentation/tractogram_registration.html

Applying transformation to tractogram can lead to invalid streamlines (out of
the bounding box), three strategies are available:
1) default, crash at saving if invalid streamlines are present
2) --keep_invalid, save invalid streamlines. Leave it to the user to run
    scil_remove_invalid_streamlines.py if needed.
3) --remove_invalid, automatically remove invalid streamlines before saving.
    Should not remove more than a few streamlines.
4) --cut_invalid, automatically cut invalid streamlines before saving.
"""

import argparse
import os
import shutil

from dipy.io.stateful_tractogram import Space, Origin, StatefulTractogram
from dipy.io.utils import create_nifti_header, get_reference_info
import h5py
import nibabel as nib
import numpy as np

from scilpy.io.streamlines import reconstruct_streamlines_from_hdf5
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)
from scilpy.utils.streamlines import transform_warp_streamlines


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_hdf5',
                   help='Path of the tractogram to be transformed.')
    p.add_argument('in_target_file',
                   help='Path of the reference target file (.trk or .nii).')
    p.add_argument('in_transfo',
                   help='Path of the file containing the 4x4 \n'
                        'transformation, matrix (.txt, .npy or .mat).')
    p.add_argument('out_hdf5',
                   help='Output tractogram filename (transformed data).')

    p.add_argument('--inverse', action='store_true',
                   help='Apply the inverse linear transformation.')
    p.add_argument('--in_deformation',
                   help='Path to the file containing a deformation field.')

    invalid = p.add_mutually_exclusive_group()
    invalid.add_argument('--cut_invalid', action='store_true',
                         help='Cut invalid streamlines rather than removing '
                              'them.\nKeep the longest segment only.\n'
                              'By default, invalid streamline are removed.')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_hdf5, args.in_target_file,
                                 args.in_transfo], args.in_deformation)
    assert_outputs_exist(parser, args, args.out_hdf5)

    # HDF5 will not overwrite the file
    if os.path.isfile(args.out_hdf5):
        os.remove(args.out_hdf5)

    in_hdf5_file = h5py.File(args.in_hdf5, 'r')
    shutil.copy(args.in_hdf5, args.out_hdf5)
    out_hdf5_file = h5py.File(args.out_hdf5, 'a')
    transfo = load_matrix_in_any_format(args.in_transfo)

    deformation_data = None
    if args.in_deformation is not None:
        deformation_data = np.squeeze(nib.load(
            args.in_deformation).get_fdata(dtype=np.float32))
    target_img = nib.load(args.in_target_file)

    for key in in_hdf5_file.keys():
        affine = in_hdf5_file.attrs['affine']
        dimensions = in_hdf5_file.attrs['dimensions']
        voxel_sizes = in_hdf5_file.attrs['voxel_sizes']
        streamlines = reconstruct_streamlines_from_hdf5(in_hdf5_file, key)

        header = create_nifti_header(affine, dimensions, voxel_sizes)
        moving_sft = StatefulTractogram(streamlines, header, Space.VOX,
                                        origin=Origin.TRACKVIS)

        new_sft = transform_warp_streamlines(moving_sft, transfo, target_img,
                                             inverse=args.inverse,
                                             deformation_data=deformation_data,
                                             remove_invalid=not args.cut_invalid,
                                             cut_invalid=args.cut_invalid)
        new_sft.to_vox()
        new_sft.to_corner()

        affine, dimensions, voxel_sizes, voxel_order = get_reference_info(target_img)
        out_hdf5_file.attrs['affine'] = affine
        out_hdf5_file.attrs['dimensions'] = dimensions
        out_hdf5_file.attrs['voxel_sizes'] = voxel_sizes
        out_hdf5_file.attrs['voxel_order'] = voxel_order

        group = out_hdf5_file[key]
        del group['data']
        group.create_dataset('data', data=new_sft.streamlines.get_data())
        del group['offsets']
        group.create_dataset('offsets', data=new_sft.streamlines._offsets)
        del group['lengths']
        group.create_dataset('lengths', data=new_sft.streamlines._lengths)

    in_hdf5_file.close()
    out_hdf5_file.close()


if __name__ == "__main__":
    main()
