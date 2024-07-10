#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script is made to fix DSI-Studio or Startrack TRK file (unknown space /
convention) to make it compatible with TrackVis, MI-Brain, Dipy Horizon
(Stateful Tractogram).

The conversion may result in invalid streamlines. By default, the script will
raise an error. You may choose to keep invalid streamlines (--no_bbox_check),
discard them or cut them. We recommand the --cut_invalid to remove invalid
points of streamlines rather removing entire streamlines.

DSI-Studio
==========

Ref: https://dsi-studio.labsolver.org/
The script will create a new stateful tractogram using the --in_dsi_fa
reference in order to fix the missing information in the header of the trk.
Will flip the x and y axes to change from LPS -> RAS convention in voxel space,
and add a 0.5 shift to the origin.

Then, if --in_native_fa is provided, will move back the tractogram to native
DWI space through registration.

Since DSI-Studio sometimes leaves some skull around the brain, the --auto_crop
aims to stabilize registration. If this option fails, manually BET both FA.
Registration is more robust at resolution above 2mm (iso), be careful.

If you are fixing many files, use this script once with --save_transfo and
verify results. Once satisfied, call the scripts on all files with
--load_transfo to save computation.

Startrack
==========

Ref: https://www.mr-startrack.com/
The script will create a new stateful tractogram using the reference in
order to fix the missing information in the header of the trk. Will flip the
x-axis.


WARNING: This script is not fully tested, DSI-Studio and Startrack evolve
quickly and results may vary depending on the data itself as well as
DSI-studio / Startrack version. But it was tested on various datasets and
worked on all of them. However, always verify the results and if a specific
case does not work. Open an issue on the Scilpy GitHub repository.

Formerly: scil_fix_dsi_studio_trk.py
"""

import argparse
import logging

from dipy.align.imaffine import (transform_centers_of_mass,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import RigidTransform3D
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import get_reference_info
from dipy.io.streamline import save_tractogram, load_tractogram
import nibabel as nib
import numpy as np

from scilpy.image.volume_operations import mask_data_with_default_cube
from scilpy.io.utils import (add_bbox_arg,
                             add_verbose_arg,
                             add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tractograms.tractogram_operations import (flip_sft,
                                                      transform_warp_sft,
                                                      get_axis_flip_vector)
from scilpy.tractograms.streamline_operations import cut_invalid_streamlines

softwares = ['dsi_studio', 'startrack']


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file from DSI studio '
                        '(.trk).')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file.')
    p.add_argument('--software', metavar='string', choices=softwares,
                   required=True,
                   help='Software used to create in_tractogram.\n'
                        'Choices: {}'.format(softwares))

    invalid = p.add_mutually_exclusive_group()
    add_bbox_arg(invalid)
    invalid.add_argument('--cut_invalid', action='store_true',
                         help='Cut invalid streamlines. Keep the longest '
                              'segment only.')
    invalid.add_argument('--remove_invalid', action='store_true',
                         help='Remove the streamlines landing out of the '
                              'bounding box.')

    g1 = p.add_argument_group(title='DSI options')
    g1.add_argument('--in_dsi_fa',
                    help='Path of the input FA from DSI Studio (.nii.gz).'
                         'Required for dsi_studio software.')
    g1.add_argument('--in_native_fa',
                    help='Path of the input FA from Dipy/MRtrix (.nii.gz).\n'
                         'If provided, move the tractogram back to a "proper" '
                         'space (includes registration).')
    g1.add_argument('--auto_crop', action='store_true',
                    help='If both FA files are not already BET, perform '
                         'registration \nusing a centered-cube crop to ignore '
                         'the skull.\nA good BET for both is more robust.')
    transfo = g1.add_mutually_exclusive_group()
    transfo.add_argument('--save_transfo', metavar='FILE',
                         help='Save estimated transformation to avoid '
                              'recomputing (.txt).')
    transfo.add_argument('--load_transfo', metavar='FILE',
                         help='Load estimated transformation to apply '
                              'to other files (.txt).')

    g2 = p.add_argument_group(title='StarTrack options')
    g2.add_argument('--reference',
                    help='Reference anatomy (.nii or .nii.gz).')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    logging.warning("This script is not fully tested, DSI-Studio and "
                    "Startrack evolve quickly and results may vary depending "
                    "on the data itself as well as DSI-studio/Startrack "
                    "version.")
    assert_outputs_exist(parser, args, args.out_tractogram)

    if args.software == 'startrack':
        if (args.in_dsi_fa or args.in_native_fa or args.auto_crop or
                args.save_transfo or args.load_transfo):
            parser.error("For the startrack software, please only set the "
                         "--reference option.")
        if args.reference is None:
            parser.error("For the startrack software, please set the "
                         "--reference option.")
        assert_inputs_exist(parser, [args.in_tractogram, args.reference])
        sft = load_tractogram(args.in_tractogram, 'same',
                              bbox_valid_check=args.bbox_check,
                              trk_header_check=False)
        sft = StatefulTractogram(sft.streamlines, args.reference, Space.VOX)

        # Startrack flips the TRK
        flip_axis = ['x']
        sft.to_vox()
        sft.streamlines._data -= get_axis_flip_vector(flip_axis)  # --------------------_> HEin?
        sft = flip_sft(sft, flip_axis)

    else:  # args.software == 'dsi_studio':
        if args.reference is not None:
            parser.error("--reference should only be set with the startrack "
                         "software.")
        if args.in_dsi_fa is None:
            parser.error("For the dsi_studio software, please provide "
                         "minimally the --in_dsi_fa option.")
        if args.load_transfo and args.in_native_fa is None:
            parser.error('When loading a transformation, the final '
                         'reference is needed, use --in_native_fa.')

        assert_inputs_exist(parser, [args.in_tractogram, args.in_dsi_fa],
                            optional=[args.load_transfo, args.in_native_fa])

        sft = load_tractogram(args.in_tractogram, 'same',
                              bbox_valid_check=args.bbox_check)

        # LPS -> RAS convention in voxel space
        sft.to_vox()
        flip_axis = ['x', 'y']
        sft = StatefulTractogram(sft.streamlines, args.in_dsi_fa, Space.VOXMM)
        sft.to_vox()
        sft.streamlines._data -= get_axis_flip_vector(flip_axis)  # --------------------_> HEin?

        sft = flip_sft(sft, flip_axis)

        # Fix origin.
        sft.to_rasmm()
        sft.streamlines._data -= [0.5, 0.5, -0.5]

        if args.in_native_fa:
            logging.info("Preparing for registration.")
            static_img = nib.load(args.in_native_fa)
            static_data = static_img.get_fdata()
            moving_img = nib.load(args.in_dsi_fa)
            moving_data = moving_img.get_fdata()

            # DSI-Studio flips the volume without changing the affine (I think)
            # So this has to be reversed (not the same problem as above)
            # Flipping again the SFT to fit with volume.
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
            sft = flip_sft(sft, flip_axis)

            if args.load_transfo:
                transfo = np.loadtxt(args.load_transfo)
            else:
                logging.info("Computing the transformation.")
                # Sometimes DSI studio has quite a lot of skull left
                # Dipy Median Otsu does not work with FA/GFA
                if args.auto_crop:
                    moving_data = mask_data_with_default_cube(moving_data)
                    static_data = mask_data_with_default_cube(static_data)

                # Since DSI Studio register to AC/PC and does not save the
                # transformation We must estimate the transformation,
                # since it's rigid it is 'easy'
                c_of_mass = transform_centers_of_mass(static_data,
                                                      static_img.affine,
                                                      moving_data,
                                                      moving_img.affine)

                nbins = 32
                sampling_prop = None
                level_iters = [1000, 100, 10]
                sigmas = [3.0, 2.0, 1.0]
                factors = [3, 2, 1]
                metric = MutualInformationMetric(nbins, sampling_prop)
                affreg = AffineRegistration(metric=metric,
                                            level_iters=level_iters,
                                            sigmas=sigmas,
                                            factors=factors)
                transform = RigidTransform3D()
                params0 = None
                rigid = affreg.optimize(static_data, moving_data, transform,
                                        params0, static_img.affine,
                                        moving_img.affine,
                                        starting_affine=c_of_mass.affine)
                transfo = rigid.affine
                if args.save_transfo:
                    np.savetxt(args.save_transfo, transfo)

            logging.info("Applying the transformation.")
            sft = transform_warp_sft(sft, transfo, static_img, inverse=True)

    if args.cut_invalid:
        sft, _ = cut_invalid_streamlines(sft)
    elif args.remove_invalid:
        sft.remove_invalid_streamlines()

    save_tractogram(sft, args.out_tractogram,
                    bbox_valid_check=args.bbox_check)


if __name__ == "__main__":
    main()
