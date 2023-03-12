#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use multiple bundles to perform a voxel-wise vote (occurence across input).
If streamlines originate from the same tractogram, streamline-wise vote
is available.

Useful to generate an average representation from bundles of a given population
or multiple bundle segmentations (gold standard).

Input tractograms must have identical header.
"""


import argparse

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible, get_reference_info
import nibabel as nib
import numpy as np
from scipy.sparse import dok_matrix

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractograms.tractogram_operations import (
    intersection_robust, union_robust)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('in_bundles', nargs='+',
                   help='Input bundles filename.')

    p.add_argument('--ratio_streamlines', type=float, default=0.5,
                   help='Minimum vote to be considered for streamlines '
                   '[%(default)s].')
    p.add_argument('--ratio_voxels', type=float, default=0.5,
                   help='Minimum vote to be considered for voxels'
                   ' [%(default)s].')

    p.add_argument('--same_tractogram', action='store_true',
                   help='All bundles need to come from the same tractogram,\n'
                        'will generate a voting for streamlines too.')
    p.add_argument('--output_prefix', default='voting_',
                   help='Output prefix, [%(default)s].')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundles)
    output_streamlines_filename = '{}streamlines.trk'.format(
        args.output_prefix)
    output_voxels_filename = '{}voxels.nii.gz'.format(args.output_prefix)
    assert_outputs_exist(parser, args, [output_voxels_filename,
                                        output_streamlines_filename])

    if not 0 <= args.ratio_voxels <= 1 or not 0 <= args.ratio_streamlines <= 1:
        parser.error('Ratios must be between 0 and 1.')

    fusion_streamlines = []
    if args.reference:
        reference_file = args.reference
    else:
        reference_file = args.in_bundles[0]
    sft_list = []
    for name in args.in_bundles:
        tmp_sft = load_tractogram_with_reference(parser, args, name)
        tmp_sft.to_vox()
        tmp_sft.to_corner()

        if not is_header_compatible(reference_file, tmp_sft):
            raise ValueError('Headers are not compatible.')
        sft_list.append(tmp_sft)
        fusion_streamlines.append(tmp_sft.streamlines)

    fusion_streamlines, _ = union_robust(fusion_streamlines)

    transformation, dimensions, _, _ = get_reference_info(reference_file)
    volume = np.zeros(dimensions)
    streamlines_vote = dok_matrix((len(fusion_streamlines),
                                   len(args.in_bundles)))

    for i in range(len(args.in_bundles)):
        sft = sft_list[i]
        binary = compute_tract_counts_map(sft.streamlines, dimensions)
        volume[binary > 0] += 1

        if args.same_tractogram:
            _, indices = intersection_robust([fusion_streamlines,
                                              sft.streamlines])
            streamlines_vote[list(indices), [i]] += 1

    if args.same_tractogram:
        real_indices = []
        ratio_value = int(args.ratio_streamlines*len(args.in_bundles))
        real_indices = np.where(np.sum(streamlines_vote,
                                       axis=1) >= ratio_value)[0]
        new_sft = StatefulTractogram.from_sft(fusion_streamlines[real_indices],
                                              sft_list[0])
        save_tractogram(new_sft, output_streamlines_filename)

    volume[volume < int(args.ratio_voxels*len(args.in_bundles))] = 0
    volume[volume > 0] = 1
    nib.save(nib.Nifti1Image(volume.astype(np.uint8), transformation),
             output_voxels_filename)


if __name__ == "__main__":
    main()
