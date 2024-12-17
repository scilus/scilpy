#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given a bundle and MRDS metrics, compute the fixel-specific
metrics at each voxel intersected by the bundle. Intersected voxels are
found by computing the intersection between the voxel grid and each streamline
in the input tractogram.

This script behaves like scil_bundle_mean_fixel_afd.py for fODFs,
but here for MRDS metrics. These latest distributions add the unique
possibility to capture fixel-based fractional anisotropy (fixel-FA), mean
diffusivity (fixel-MD), radial diffusivity (fixel-RD) and
axial diffusivity (fixel-AD).

Fixel-specific metrics are metrics extracted from
Multi-Resolution Discrete-Search (MRDS) solutions.
There are as many values per voxel as there are fixels extracted. The
values chosen for a given voxel is the one belonging to the lobe better aligned
with the current streamline segment.

Input files come from scil_mrds_metrics.py command.

Output metrics will be named: [prefix]_mrds_[metric_name].nii.gz

Please use a bundle file rather than a whole tractogram.
"""

import argparse

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_headers_compatible,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.tractanalysis.mrds_along_streamlines \
    import mrds_metrics_along_streamlines
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)
    
    p.add_argument('in_bundle',
                   help='Path of the bundle file.')
    p.add_argument('in_pdds',
                   help='Path of the MRDS PDDs volume.')

    g = p.add_argument_group(title='MRDS metrics input')
    g.add_argument('--fa',
                   help='Path of the fixel-specific metric FA volume.')
    g.add_argument('--md',
                   help='Path of the fixel-specific metric MD volume.')
    g.add_argument('--rd',
                   help='Path of the fixel-specific metric RD volume.')
    g.add_argument('--ad',
                   help='Path of the fixel-specific metric AD volume.')

    p.add_argument('--prefix', default='result',
                   help='Prefix of the MRDS fixel results.')

    p.add_argument('--length_weighting', action='store_true',
                   help='If set, will weight the values according to '
                        'segment lengths. [%(default)s]')

    p.add_argument('--max_theta', default=60, type=float,
                   help='Maximum angle (in degrees) condition on fixel '
                        'alignment. [%(default)s]')

    add_reference_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_metrics = []
    out_metrics = []
    if args.fa is not None:
        in_metrics.append(args.fa)
        out_metrics.append('{}_mrds_fFA.nii.gz'.format(args.prefix))
    if args.ad is not None:
        in_metrics.append(args.ad)
        out_metrics.append('{}_mrds_fAD.nii.gz'.format(args.prefix))
    if args.rd is not None:
        in_metrics.append(args.rd)
        out_metrics.append('{}_mrds_fRD.nii.gz'.format(args.prefix))
    if args.md is not None:
        in_metrics.append(args.md)
        out_metrics.append('{}_mrds_fMD.nii.gz'.format(args.prefix))

    if in_metrics == []:
        parser.error('At least one metric is required.')

    assert_inputs_exist(parser, [args.in_bundle,
                                 args.in_pdds], in_metrics)
    assert_headers_compatible(parser, [args.in_bundle, args.in_pdds],
                              in_metrics)

    assert_outputs_exist(parser, args, out_metrics)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    pdds_img = nib.load(args.in_pdds)
    affine = pdds_img.affine
    header = pdds_img.header

    in_metrics_data = [nib.load(metric).get_fdata(dtype=np.float32) for metric in in_metrics]
    fixel_metrics =\
        mrds_metrics_along_streamlines(sft,
                                       pdds_img.get_fdata(dtype=np.float32),
                                       in_metrics_data,
                                       args.max_theta,
                                       args.length_weighting)

    for metric_id, curr_metric in enumerate(fixel_metrics):
        nib.Nifti1Image(curr_metric.astype(np.float32),
                        affine=affine,
                        header=header,
                        dtype=np.float32).to_filename(out_metrics[metric_id])


if __name__ == '__main__':
    main()
