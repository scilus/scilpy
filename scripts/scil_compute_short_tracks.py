#!/usr/bin/env python3
import argparse
import nibabel as nib
import numpy as np

from scilpy.io.utils import add_overwrite_arg, add_sh_basis_args
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.utils import find_order_from_nb_coeff
from scilpy.tracking.utils import add_seeding_options
from scilpy.tracking.short_tracks import track_short_tracks
from dipy.tracking.utils import random_seeds_from_mask
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import save_tractogram


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    # mandatory tracking options
    p.add_argument('in_odf',
                   help='File containing the orientation diffusion function \n'
                        'as spherical harmonics file (.nii.gz). Ex: ODF or '
                        'fODF.')
    p.add_argument('in_mask',
                   help='Tracking mask (.nii.gz).\n'
                        'Tracking will stop outside this mask.')
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')

    add_seeding_options(p)
    p.add_argument('--step_size', type=float, default=0.5,
                   help='Step size in mm. [0.5]')
    p.add_argument('--theta', type=float, default=20.0,
                   help='Maximum angle between 2 steps. [20.0]')
    p.add_argument('--min_length', type=float, default=10.0,
                   help='Minimum length of the streamline in mm. [10.0]')
    p.add_argument('--max_length', type=float, default=20.0,
                   help='Maximum length of the streamline in mm. [20.0]')
    p.add_argument('--batch_size', type=int, default=100000,
                   help='Approximate size of GPU batches. [100000]')
    add_overwrite_arg(p)
    add_sh_basis_args(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    odf_sh_img = nib.load(args.in_odf)
    mask = get_data_as_mask(nib.load(args.in_mask))

    if args.npv:
        nb_seeds = args.npv
        seed_per_vox = True
    elif args.nt:
        nb_seeds = args.nt
        seed_per_vox = False
    else:
        nb_seeds = 1
        seed_per_vox = True

    voxel_size = odf_sh_img.header.get_zooms()[0]

    vox_step_size = args.step_size / voxel_size
    seeds = random_seeds_from_mask(
        mask, np.eye(4),
        seeds_count=nb_seeds,
        seed_count_per_voxel=seed_per_vox,
        random_seed=None)

    odf_sh = odf_sh_img.get_fdata()
    order = find_order_from_nb_coeff(odf_sh)
    vox_max_length = args.max_length / voxel_size
    vox_min_length = args.min_length / voxel_size
    streamlines = track_short_tracks(odf_sh_img.get_fdata(), seeds, mask,
                                     vox_step_size, vox_min_length,
                                     vox_max_length, args.theta,
                                     args.batch_size, order,
                                     args.sh_basis)

    if True:
        # visualize output
        from fury import window, actor
        line_actor = actor.line(streamlines)
        s = window.Scene()
        s.add(line_actor)
        window.show(s)

    sft = StatefulTractogram(streamlines, odf_sh_img, Space.VOX,
                             Origin.TRACKVIS)
    save_tractogram(sft, args.out_tractogram)


if __name__ == '__main__':
    main()
