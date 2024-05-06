#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script is designed to compare and help visualize differences between
two tractograms. This can be especially useful in studies where multiple
tractograms from different algorithms or parameters need to be compared.

A similar script (scil_bundle_pairwise_comparison.py) is available for bundles,
with metrics more adapted to bundles (and spatial agreement).

The difference is computed in terms of
- A voxel-wise spatial distance between streamlines crossing each voxel.
    This can help to see if both tractography reconstructions at each voxel
    look similar (out_diff.nii.gz)
- An angular correlation (ACC) between streamline orientation from TODI.
    This compares the local orientation of streamlines at each voxel
    (out_acc.nii.gz)
- A patch-wise correlation between streamline density maps from both
    tractograms. This compares where the high/low density regions agree or not
    (out_corr.nii.gz)
- A heatmap combining all the previous metrics using a harmonic means of the
    normalized metrics to summarize general agreement (out_heatmap.nii.gz)
"""

import argparse
import logging
import os

import nibabel as nib

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             add_processes_arg,
                             add_verbose_arg,
                             assert_headers_compatible,
                             validate_nbr_processes)
from scilpy.tractanalysis.reproducibility_measures import \
    tractogram_pairwise_comparison


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram_1',
                   help='Input tractogram 1.')
    p.add_argument('in_tractogram_2',
                   help='Input tractogram 2.')

    p.add_argument('--out_dir', default='',
                   help='Directory where all output files will be saved. '
                        '\nIf not specified, outputs will be saved in the '
                        'current directory.')
    p.add_argument('--out_prefix', default='out',
                   help='Prefix for output files. Useful for distinguishing '
                        'between different runs [%(default)s].')

    p.add_argument('--in_mask', metavar='IN_FILE',
                   help='Optional input mask.')
    p.add_argument('--skip_streamlines_distance', action='store_true',
                   help='Skip computation of the spatial distance between '
                        'streamlines. Slowest part of the computation.')
    add_processes_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_tractogram_1, args.in_tractogram_2],
                        [args.in_mask, args.reference])
    to_verify = [args.in_tractogram_1, args.in_tractogram_2]
    if args.in_mask:
        to_verify.append(args.in_mask)
    assert_headers_compatible(parser, to_verify, reference=args.reference)

    if args.out_prefix and args.out_prefix[-1] == '_':
        args.out_prefix = args.out_prefix[:-1]
    out_corr_filename = os.path.join(
        args.out_dir, '{}_correlation.nii.gz'.format(args.out_prefix))
    out_acc_filename = os.path.join(
        args.out_dir, '{}_acc.nii.gz'.format(args.out_prefix))
    out_diff_filename = os.path.join(
        args.out_dir, '{}_diff.nii.gz'.format(args.out_prefix))
    out_merge_filename = os.path.join(
        args.out_dir, '{}_heatmap.nii.gz'.format(args.out_prefix))
    assert_output_dirs_exist_and_empty(parser, args, [], optional=args.out_dir)
    assert_outputs_exist(parser, args, [out_corr_filename,
                                        out_acc_filename,
                                        out_merge_filename],
                         out_diff_filename)
    nbr_cpu = validate_nbr_processes(parser, args)

    logging.info('Loading tractograms...')
    sft_1 = load_tractogram_with_reference(parser, args, args.in_tractogram_1)
    sft_2 = load_tractogram_with_reference(parser, args, args.in_tractogram_2)
    mask = nib.load(args.in_mask) if args.in_mask else None

    acc_data, corr_data, diff_data, heatmap, _ = \
        tractogram_pairwise_comparison(sft_1, sft_2, mask, nbr_cpu,
                                       args.skip_streamlines_distance)

    logging.info('Saving results...')
    nib.save(nib.Nifti1Image(acc_data, sft_1.affine), out_acc_filename)
    nib.save(nib.Nifti1Image(corr_data, sft_1.affine), out_corr_filename)
    if not args.skip_streamlines_distance:
        nib.save(nib.Nifti1Image(diff_data, sft_1.affine), out_diff_filename)
    nib.save(nib.Nifti1Image(heatmap, sft_1.affine), out_merge_filename)


if __name__ == "__main__":
    main()
