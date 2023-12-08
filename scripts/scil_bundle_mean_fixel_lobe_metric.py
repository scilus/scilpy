#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given a bundle and Bingham coefficients, compute the average lobe-specific
metric at each voxel intersected by the bundle. Intersected voxels are
found by computing the intersection between the voxel grid and each streamline
in the input tractogram.

This script behaves like scil_compute_mean_fixel_afd_from_bundles.py for fODFs,
but here for Bingham distributions. These latest distributions add the unique
possibility to capture fixel-based fiber spread (FS) and fiber fraction (FF).
FD from the bingham should be "equivalent" to the AFD_fixel we are used to.

Bingham coefficients volume must come from scil_fodf_to_bingham.py
and lobe-specific metrics comes from scil_fodf_lobe_specific_metrics.py.

Lobe-specific metrics are metrics extracted from Bingham distributions fitted
to fODF. There are as many values per voxel as there are lobes extracted. The
values chosen for a given voxelis the one belonging to the lobe better aligned
with the current streamline segment.

Please use a bundle file rather than a whole tractogram.
"""

import argparse

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.tractanalysis.lobe_metrics_along_streamlines \
    import lobe_specific_metric_map_along_streamlines


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Path of the bundle file.')
    p.add_argument('in_bingham',
                   help='Path of the Bingham volume.')
    p.add_argument('in_lobe_metric',
                   help='Path of the lobe-specific metric (FD, FS, or FF) '
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
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bundle,
                                 args.in_bingham,
                                 args.in_lobe_metric])
    assert_outputs_exist(parser, args, [args.out_mean_map])

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    bingham_img = nib.load(args.in_bingham)
    metric_img = nib.load(args.in_lobe_metric)

    if bingham_img.shape[-2] != metric_img.shape[-1]:
        parser.error('Dimension mismatch between Bingham coefficients '
                     'and lobe-specific metric image.')

    metric_mean_map =\
        lobe_specific_metric_map_along_streamlines(sft,
                                                   bingham_img.get_fdata(),
                                                   metric_img.get_fdata(),
                                                   args.max_theta,
                                                   args.length_weighting)

    nib.Nifti1Image(metric_mean_map.astype(np.float32),
                    bingham_img.affine).to_filename(args.out_mean_map)


if __name__ == '__main__':
    main()
