#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the mean Apparent Fiber Density (AFD) and mean Radial fODF (radfODF)
maps along a bundle.

This is the "real" fixel-based fODF amplitude along every streamline
of the bundle provided, averaged at every voxel.

Please use a bundle file rather than a whole tractogram.
"""

import argparse

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             add_reference_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.reconst.bingham_metrics_along_streamlines \
    import fiber_density_map_along_streamlines

EPILOG = """
Reference:
    [1] Raffelt, D., Tournier, JD., Rose, S., Ridgway, GR., Henderson, R.,
        Crozier, S., Salvado, O., & Connelly, A. (2012).
        Apparent Fibre Density: a novel measure for the analysis of
        diffusion-weighted magnetic resonance images. NeuroImage, 59(4),
        3976-3994.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Path of the bundle file.')
    p.add_argument('in_bingham',
                   help='Path of the Bingham volume.')
    p.add_argument('in_fd',
                   help='Path of the fiber density volume.')
    p.add_argument('fd_mean_map',
                   help='Path of the output mean fiber density map.')

    p.add_argument('--length_weighting', action='store_true',
                   help='If set, will weigh the FD values according to '
                        'segment lengths. [%(default)s]')

    add_reference_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bundle, args.in_bingham, args.in_fd])
    assert_outputs_exist(parser, args, [args.fd_mean_map])

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    bingham_img = nib.load(args.in_bingham)
    fd_img = nib.load(args.in_fd)

    if bingham_img.shape[-1] % fd_img.shape[-1] != 0:
        parser.error('Dimension mismatch between Bingham coefficients '
                     'and fiber density image.')

    fd_mean_map = fiber_density_map_along_streamlines(sft,
                                                      bingham_img.get_fdata(),
                                                      fd_img.get_fdata())

    nib.Nifti1Image(fd_mean_map.astype(np.float32),
                    bingham_img.affine).to_filename(args.fd_mean_map)


if __name__ == '__main__':
    main()
