#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given a bundle and Bingham coefficients, compute the average Bingham
metric at each voxel intersected by the bundle. Intersected voxels are
found by computing the intersection between the voxel grid and each streamline
in the input tractogram.

This script behaves like scil_bundle_mean_fixel_afd for fODFs,
but here for Bingham distributions. These add the unique possibility to capture
fixel-based fiber spread (FS) and fiber fraction (FF). FD from the bingham
should be "equivalent" to the AFD_fixel we are used to.

Bingham coefficients volume must come from scil_fodf_to_bingham
and Bingham metrics comes from scil_bingham_metrics.

Bingham metrics are extracted from Bingham distributions fitted to fODF. There
are as many values per voxel as there are lobes extracted. The values chosen
for a given voxelis the one belonging to the lobe better aligned with the
current streamline segment.

Please use a bundle file rather than a whole tractogram.

"""

import argparse
import logging

import numpy as np

from scilpy.io.stateful_image import StatefulImage
from scilpy.io.streamlines import (load_tractogram_with_reference,
                                   rebind_sft_to_simg)
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, assert_headers_compatible)
from scilpy.tractanalysis.bingham_metric_along_streamlines \
    import bingham_metric_map_along_streamlines
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_bundle',
                   help='Path of the bundle file.')
    p.add_argument('in_bingham',
                   help='Path of the Bingham volume.')
    p.add_argument('in_bingham_metric',
                   help='Path of the Bingham metric (FD, FS, or FF) '
                        'volume.')
    p.add_argument('out_mean_map',
                   help='Path of the output mean map.')

    p.add_argument('--length_weighting', action='store_true',
                   help='If set, will weigh the FD values according to '
                        'segment lengths.')

    p.add_argument('--max_theta', default=60, type=float,
                   help='Maximum angle (in degrees) condition on lobe '
                        'alignment. [%(default)s]')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_bundle, args.in_bingham,
                                 args.in_bingham_metric],
                        args.reference)
    assert_outputs_exist(parser, args, [args.out_mean_map])
    assert_headers_compatible(parser, [args.in_bundle, args.in_bingham,
                                       args.in_bingham_metric],
                              reference=args.reference)

    # Bingham mu1/mu2 are directional: reorient the grid to RAS and convert
    # the direction vectors from world to voxel space.
    bingham_simg = StatefulImage.load(args.in_bingham, is_orientation=True)
    bingham_simg.to_ras()
    bingham_coeffs = bingham_simg.to_voxel_direction()

    # The metric map (FD/FS/FF) is scalar, just needs the matching grid.
    metric_simg = StatefulImage.load(args.in_bingham_metric)
    metric_simg.to_ras()
    metric_data = metric_simg.get_fdata()

    if bingham_coeffs.shape[-2] != metric_data.shape[-1]:
        parser.error('Dimension mismatch between Bingham coefficients '
                     'and Bingham metric image.')

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft = rebind_sft_to_simg(sft, bingham_simg)

    metric_mean_map =\
        bingham_metric_map_along_streamlines(sft,
                                             bingham_coeffs,
                                             metric_data,
                                             args.max_theta,
                                             args.length_weighting)

    StatefulImage.create_from(
        metric_mean_map.astype(np.float32), bingham_simg).save(
        args.out_mean_map)


if __name__ == '__main__':
    main()
