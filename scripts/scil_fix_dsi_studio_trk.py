#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script is made to fix DSI-Studio TRK file (unknown space/convention) to
make it compatible with TrackVis, MI-Brain, Dipy Horizon (Stateful Tractogram).

The script either make it match with an anatomy from DSI-Studio (AC-PC aligned,
sometimes flipped) or if --in_native_fa is provided it moves it back to native
DWI space (this involved registration).

Since DSI-Studio sometimes leaves some skull around the brain, the --auto_crop
aims to stabilize registration. If this option fails, manually BET both FA.
Registration is more robust at resolution above 2mm (iso), be careful.

If you are fixing bundles, use this script once with --save_transfo and verify
results. Once satisfied, call the scripts on bundles using a bash for loop with
--load_transfo to save computation.

We recommand the --cut_invalid to remove invalid points of streamlines rather
removing entire streamlines.

This script was tested on various datasets and worked on all of them. However,
always verify the results and if a specific case does not work. Open an issue
on the Scilpy GitHub repository.

WARNING: This script is still experimental, DSI-Studio evolves quickly and
results may vary depending on the data itself as well as DSI-studio version.
"""

import argparse

from dipy.align.imaffine import (transform_centers_of_mass,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import RigidTransform3D
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import get_reference_info
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.reconst.utils import _roi_in_volume, _mask_from_roi
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.streamlines import (transform_warp_sft,
                                      cut_invalid_streamlines)
from scilpy.utils.transformation import flip_sft


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dsi_tractogram',
                   help='Path of the input tractogram file from DSI studio '
                        '(.trk).')
    p.add_argument('in_dsi_fa',
                   help='Path of the input FA from DSI Studio (.nii.gz).')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file.')
    p.add_argument('--in_native_fa',
                   help='Path of the input FA from Dipy/MRtrix (.nii.gz).\n'
                        'Move the tractogram back to a "proper" space, include'
                        'registration.')
    p.add_argument('--auto_crop', action='store_true',
                   help='If both FA are not already BET, perform registration \n'
                        'using a centered-cube crop to ignore the skull.\n'
                        'A good BET for both is more robust.')
    transfo = p.add_mutually_exclusive_group()
    transfo.add_argument('--save_transfo', metavar='FILE',
                         help='Save estimated transformation to avoid '
                              'recomputing (.txt).')
    transfo.add_argument('--load_transfo', metavar='FILE',
                         help='Load estimated transformation to apply to other '
                              'files (.txt).')
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
    add_overwrite_arg(p)

    return p


def get_axis_shift_vector(flip_axes):
    shift_vector = np.zeros(3)
    if 'x' in flip_axes:
        shift_vector[0] = -1.0
    if 'y' in flip_axes:
        shift_vector[1] = -1.0
    if 'z' in flip_axes:
        shift_vector[2] = -1.0

    return shift_vector


def cube_crop_data(data):
    shape = np.array(data.shape[:3])
    roi_center = shape // 2
    roi_radii = _roi_in_volume(shape, roi_center, shape // 3)
    roi_mask = _mask_from_roi(shape, roi_center, roi_radii)

    return data * roi_mask


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.load_transfo and args.in_native_fa is None:
        parser.error('When loading a transformation, the final reference is '
                     'needed, use --in_native_fa.')
    assert_inputs_exist(parser, [args.in_dsi_tractogram, args.in_dsi_fa],
                        optional=args.in_native_fa)
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram(args.in_dsi_tractogram, 'same',
                          bbox_valid_check=False)

    # LPS -> RAS convention in voxel space
    sft.to_vox()
    flip_axis = ['x', 'y']
    sft_fix = StatefulTractogram(sft.streamlines, args.in_dsi_fa,
                                 Space.VOXMM)
    sft_fix.to_vox()
    sft_fix.streamlines._data -= get_axis_shift_vector(flip_axis)

    sft_flip = flip_sft(sft_fix, flip_axis)

    sft_flip.to_rasmm()
    sft_flip.streamlines._data -= [0.5, 0.5, -0.5]

    if not args.in_native_fa:
        if args.cut_invalid:
            sft_flip, _ = cut_invalid_streamlines(sft_flip)
        elif args.remove_invalid:
            sft_flip.remove_invalid_streamlines()
        save_tractogram(sft_flip, args.out_tractogram,
                        bbox_valid_check=not args.keep_invalid)
    else:
        static_img = nib.load(args.in_native_fa)
        static_data = static_img.get_fdata()
        moving_img = nib.load(args.in_dsi_fa)
        moving_data = moving_img.get_fdata()

        # DSI-Studio flips the volume without changing the affine (I think)
        # So this has to be reversed (not the same problem as above)
        vox_order = get_reference_info(moving_img)[3]
        flip_axis = []
        if vox_order[0] == 'L':
            moving_data = moving_data[::-1, :, :]
            flip_axis.append('x')
        if vox_order[1] == 'P':
            moving_data = moving_data[:, ::-1, :]
            flip_axis.append('y')
        if vox_order[2] == 'I':
            moving_data = moving_data[:, :, ::-1]
            flip_axis.append('z')
        sft_flip_back = flip_sft(sft_flip, flip_axis)

        if args.load_transfo:
            transfo = np.loadtxt(args.load_transfo)
        else:
            # Sometimes DSI studio has quite a lot of skull left
            # Dipy Median Otsu does not work with FA/GFA
            if args.auto_crop:
                moving_data = cube_crop_data(moving_data)
                static_data = cube_crop_data(static_data)

            # Since DSI Studio register to AC/PC and does not save the
            # transformation We must estimate the transformation, since it's
            # rigid it is 'easy'
            c_of_mass = transform_centers_of_mass(static_data, static_img.affine,
                                                  moving_data, moving_img.affine)

            nbins = 32
            sampling_prop = None
            level_iters = [1000, 100, 10]
            sigmas = [3.0, 2.0, 1.0]
            factors = [3, 2, 1]
            metric = MutualInformationMetric(nbins, sampling_prop)
            affreg = AffineRegistration(metric=metric, level_iters=level_iters,
                                        sigmas=sigmas, factors=factors)
            transform = RigidTransform3D()
            rigid = affreg.optimize(static_data, moving_data, transform, None,
                                    static_img.affine, moving_img.affine,
                                    starting_affine=c_of_mass.affine)
            transfo = rigid.affine
            if args.save_transfo:
                np.savetxt(args.save_transfo, transfo)

        new_sft = transform_warp_sft(sft_flip_back, transfo,
                                     static_img, inverse=True,
                                     remove_invalid=args.remove_invalid,
                                     cut_invalid=args.cut_invalid)

        if args.cut_invalid:
            new_sft, _ = cut_invalid_streamlines(new_sft)
        elif args.remove_invalid:
            new_sft.remove_invalid_streamlines()
        save_tractogram(new_sft, args.out_tractogram,
                        bbox_valid_check=not args.keep_invalid)


if __name__ == "__main__":
    main()
