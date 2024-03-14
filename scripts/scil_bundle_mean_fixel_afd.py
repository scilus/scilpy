#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the mean Apparent Fiber Density (AFD) and mean Radial fODF (radfODF)
maps along a bundle.

This is the "real" fixel-based fODF amplitude along every streamline
of the bundle provided, averaged at every voxel.

Please use a bundle file rather than a whole tractogram.

Formerly: scil_compute_fixel_afd_from_bundles.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             add_reference_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             parse_sh_basis_arg, assert_headers_compatible)
from scilpy.tractanalysis.afd_along_streamlines \
    import afd_map_along_streamlines

EPILOG = """
Reference:
    [1] Raffelt, D., Tournier, JD., Rose, S., Ridgway, GR., Henderson, R.,
        Crozier, S., Salvado, O., & Connelly, A. (2012).
        Apparent Fibre Density: a novel measure for the analysis of
        diffusion-weighted magnetic resonance images. NeuroImage, 59(4),
        3976--3994.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Path of the bundle file.')
    p.add_argument('in_fodf',
                   help='Path of the fODF volume in spherical harmonics (SH).')
    p.add_argument('afd_mean_map',
                   help='Path of the output mean AFD map.')

    p.add_argument('--length_weighting', action='store_true',
                   help='If set, will weigh the AFD values according to '
                        'segment lengths. [%(default)s]')

    add_reference_arg(p)
    add_sh_basis_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_bundle, args.in_fodf],
                        args.reference)
    assert_outputs_exist(parser, args, [args.afd_mean_map])
    assert_headers_compatible(parser, [args.in_bundle, args.in_fodf],
                              reference=args.reference)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    fodf_img = nib.load(args.in_fodf)

    sh_basis, is_legacy = parse_sh_basis_arg(args)

    afd_mean_map, rd_mean_map = afd_map_along_streamlines(
                                                sft,
                                                fodf_img,
                                                sh_basis,
                                                args.length_weighting,
                                                is_legacy=is_legacy)

    nib.Nifti1Image(afd_mean_map.astype(np.float32),
                    fodf_img.affine).to_filename(args.afd_mean_map)


if __name__ == '__main__':
    main()
