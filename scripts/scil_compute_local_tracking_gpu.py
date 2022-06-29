#!/usr/bin/env python3
"""
Perform probabilistic tractography on a ODF field inside a binary mask.
The tracking is executed on the GPU using the OpenCL API.

Streamlines are filtered by minimum length, but not by maximum length. For this
reason, there may be streamlines ending in the deep white matter. In order to
use the resulting tractogram for analysis, it should be cleaned with
scil_filter_tractogram_anatomically.py.

The ODF image and mask are interpolated using nearest-neighbor interpolation.

The script also incorporates ideas from Ensemble Tractography [1] (ET). Given
a list of maximum angles, a different angle drawn at random from the set will
be used for each streamline.

In order to use the script, you must have a OpenCL compatible GPU and install
the pyopencl package via `pip install pyopencl`.
"""

import argparse
import logging
from time import perf_counter
import nibabel as nib
import numpy as np

from nibabel.streamlines.tractogram import LazyTractogram, TractogramItem
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.io.image import get_data_as_mask
from scilpy.tracking.utils import (add_seeding_options,
                                   add_mandatory_options_tracking)
from scilpy.tracking.tracker import GPUTacker
from dipy.tracking.utils import random_seeds_from_mask
from dipy.tracking.streamlinespeed import compress_streamlines
from dipy.io.utils import get_reference_info, create_tractogram_header
from scilpy.io.utils import verify_compression_th


EPILOG = """
[1] Takemura, H. et al (2016). Ensemble tractography. PLoS Computational
    Biology, 12(2), e1004692.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    # rename `optional arguments` group to `Generic options`
    p._optionals.title = 'Generic options'

    # mandatory tracking options
    add_mandatory_options_tracking(p)

    track_g = p.add_argument_group('Tracking options')
    track_g.add_argument('--step_size', type=float, default=0.5,
                         help='Step size in mm. [%(default)s]')
    track_g.add_argument('--theta', type=float, nargs='+', default=20.0,
                         help='Maximum angle between 2 steps. If more than one'
                              ' value\nare given, the maximum angle will be '
                              'drawn at random\nfrom the distribution for each'
                              ' streamline. [%(default)s]')
    track_g.add_argument('--min_length', type=float, default=20.0,
                         help='Minimum length of the streamline '
                              'in mm. [%(default)s]')
    track_g.add_argument('--max_length', type=float, default=300.0,
                         help='Maximum length of the streamline '
                              'in mm. [%(default)s]')
    track_g.add_argument('--forward_only', action='store_true',
                         help='Only perform forward tracking.')
    add_sh_basis_args(track_g)

    # seeding options
    add_seeding_options(p)
    out_g = p.add_argument_group('Output options')
    out_g.add_argument('--save_seeds', action='store_true',
                       help='Save seed positions in data_per_streamline.')
    out_g.add_argument('--compress', type=float,
                       help='Compress streamlines using the given threshold.')

    # random number generator for SF sampling
    out_g.add_argument('--rng_seed', type=int,
                       help='Random number generator seed.')
    add_overwrite_arg(out_g)

    gpu_g = p.add_argument_group('GPU options')
    gpu_g.add_argument('--batch_size', type=int, default=100000,
                       help='Approximate size of GPU batches (number\n'
                            'of streamlines to track in parallel).'
                            ' [%(default)s]')

    log_g = p.add_argument_group('Logging options')
    add_verbose_arg(log_g)
    return p


def main():
    t_init = perf_counter()
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.in_odf, args.in_mask, args.in_seed])
    assert_outputs_exist(parser, args, args.out_tractogram)
    if args.compress is not None:
        verify_compression_th(args.compress)

    odf_sh_img = nib.load(args.in_odf)
    mask = get_data_as_mask(nib.load(args.in_mask))
    seed_mask = get_data_as_mask(nib.load(args.in_seed))
    odf_sh = odf_sh_img.get_fdata(dtype=np.float32)

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

    # Seeds are returned with origin `center`.
    # However, GPUTracker expects origin to be `corner`.
    # Therefore, we need to shift the seed positions by half voxel.
    seeds = random_seeds_from_mask(
        seed_mask, np.eye(4),
        seeds_count=nb_seeds,
        seed_count_per_voxel=seed_per_vox,
        random_seed=args.rng_seed) + 0.5
    logging.info('Generated {0} seed positions in {1:.2f}s.'
                 .format(len(seeds), perf_counter() - t0))

    voxel_size = odf_sh_img.header.get_zooms()[0]
    vox_step_size = args.step_size / voxel_size
    vox_max_length = args.max_length / voxel_size
    vox_min_length = args.min_length / voxel_size
    min_strl_len = int(vox_min_length / vox_step_size) + 1
    max_strl_len = int(vox_max_length / vox_step_size) + 1

    # initialize tracking
    tracker = GPUTacker(odf_sh, mask, seeds, vox_step_size, min_strl_len,
                        max_strl_len, args.theta, args.sh_basis,
                        args.batch_size, args.forward_only, args.rng_seed)

    # wrapper for tracker.track() yielding one TractogramItem per
    # streamline for use with the LazyTractogram.
    def tracks_generator_wrapper():
        for strl, seed in tracker.track():
            # seed must be saved in voxel space, with origin `center`.
            dps = {'seeds': seed - 0.5} if args.save_seeds else {}

            # TODO: Investigate why the streamline must NOT be shifted to
            # origin `corner` for LazyTractogram.
            strl *= voxel_size  # in mm.
            if args.compress:
                strl = compress_streamlines(strl, args.compress)
            yield TractogramItem(strl, dps, {})

    # instantiate tractogram
    tractogram = LazyTractogram.from_data_func(tracks_generator_wrapper)
    tractogram.affine_to_rasmm = odf_sh_img.affine

    filetype = nib.streamlines.detect_format(args.out_tractogram)
    reference = get_reference_info(odf_sh_img)
    header = create_tractogram_header(filetype, *reference)

    # Use generator to save the streamlines on-the-fly
    nib.streamlines.save(tractogram, args.out_tractogram, header=header)
    logging.info('Saved tractogram to {0}.'.format(args.out_tractogram))

    # Total runtime
    logging.info('Total runtime of {0:.2f}s.'.format(perf_counter() - t_init))


if __name__ == '__main__':
    main()
