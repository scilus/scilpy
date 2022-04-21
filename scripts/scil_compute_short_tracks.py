#!/usr/bin/env python3
"""
Perform probabilistic short-tracks tractography[1] on a ODF field inside a
binary mask. The tracking is executed on the GPU using the OpenCL API.

In short-tracks tractography, streamlines are seeded inside the tracking
mask and all have a similar length. They are not expected to connect two
regions of interest and can end inside the white matter mask.

No backward tracking is done from the seed point and the streamlines are
returned as soon as they reach maximum length. The ODF image and mask are
interpolated using nearest-neighbor interpolation.

The script also incorporates ideas from Ensemble Tractography[2] (ET). Given
a list of maximum angles, a different angle drawn at random from the set will
be used for each streamline.
"""

import argparse
import logging
from time import perf_counter
import nibabel as nib
import numpy as np

from nibabel.streamlines.tractogram import LazyTractogram
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.utils import find_order_from_nb_coeff
from scilpy.tracking.utils import add_seeding_options
from scilpy.tracking.short_tracks import track_short_tracks
from scilpy.gpuparallel.opencl_utils import have_opencl
from dipy.tracking.utils import random_seeds_from_mask
from dipy.io.utils import get_reference_info, create_tractogram_header


EPILOG = """
[1] Calamante, F. et al (2012). Super-resolution track-density imaging studies
    of mouse brain: Comparison to histology. NeuroImage, 59(1), 286-296.
[2] Takemura, H. et al (2016). Ensemble tractography. PLoS Computational
    Biology, 12(2), e1004692.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
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
                   help='Step size in mm. [%(default)s]')
    p.add_argument('--theta', type=float, nargs='+', default=20.0,
                   help='Maximum angle between 2 steps. If more than one value'
                        '\nare given, the maximum angle will be drawn at '
                        'random\nfrom the distribution for each streamline. '
                        '[%(default)s]')
    p.add_argument('--min_length', type=float, default=10.0,
                   help='Minimum length of the streamline '
                        'in mm. [%(default)s]')
    p.add_argument('--max_length', type=float, default=20.0,
                   help='Maximum length of the streamline '
                        'in mm. [%(default)s]')
    p.add_argument('--odf_sharpness', type=float, default=1.0,
                   help='Exponent on ODF amplitude to control'
                        ' sharpness. [%(default)s]')
    p.add_argument('--batch_size', type=int, default=100000,
                   help='Approximate size of GPU batches. [%(default)s]')
    p.add_argument('--save_seeds', action='store_true',
                   help='Save seed positions in data_per_streamline.')

    add_sh_basis_args(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def main():
    t_init = perf_counter()
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not have_opencl:
        raise RuntimeError('pyopencl is not installed. In order to run'
                           'the script, you need to install it first.')

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.in_odf, args.in_mask])
    assert_outputs_exist(parser, args, args.out_tractogram)

    odf_sh_img = nib.load(args.in_odf)
    mask = get_data_as_mask(nib.load(args.in_mask))
    odf_sh = odf_sh_img.get_fdata()
    order = find_order_from_nb_coeff(odf_sh)

    t0 = perf_counter()
    if args.npv:
        nb_seeds = args.npv
        seed_per_vox = True
    elif args.nt:
        nb_seeds = args.nt
        seed_per_vox = False
    else:
        nb_seeds = 1
        seed_per_vox = True

    seeds = random_seeds_from_mask(
        mask, np.eye(4),
        seeds_count=nb_seeds,
        seed_count_per_voxel=seed_per_vox,
        random_seed=None)
    logging.info('Generated {0} seed positions in {1:.2f}s.'
                 .format(len(seeds), perf_counter() - t0))

    voxel_size = odf_sh_img.header.get_zooms()[0]
    vox_step_size = args.step_size / voxel_size
    vox_max_length = args.max_length / voxel_size
    vox_min_length = args.min_length / voxel_size
    streamlines, seeds = track_short_tracks(odf_sh, seeds, mask,
                                            vox_step_size, vox_min_length,
                                            vox_max_length, args.theta,
                                            args.odf_sharpness,
                                            args.batch_size, order,
                                            args.sh_basis)

    # Save tractogram to file
    t0 = perf_counter()
    data_per_streamlines = {'seeds': lambda: seeds} if args.save_seeds else {}
    tractogram = LazyTractogram(lambda: streamlines,
                                data_per_streamline=data_per_streamlines,
                                affine_to_rasmm=odf_sh_img.affine)

    filetype = nib.streamlines.detect_format(args.out_tractogram)
    reference = get_reference_info(odf_sh_img)
    header = create_tractogram_header(filetype, *reference)

    # Use generator to save the streamlines on-the-fly
    nib.streamlines.save(tractogram, args.out_tractogram, header=header)

    logging.info('Saved tractogram to {0} in {1:.2f}s.'
                 .format(args.out_tractogram, perf_counter() - t0))

    # Total runtime
    logging.info('Total runtime of {0:.2f}s.'
                 .format(perf_counter() - t_init))


if __name__ == '__main__':
    main()
