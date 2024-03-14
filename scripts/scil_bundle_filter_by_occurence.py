#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use multiple versions of a same bundle and detect the most probable voxels by
using a threshold on the occurence, voxel-wise. With threshold 0.5, this is
a majority vote. This is useful to generate an average representation from
bundles of a given population.

If streamlines originate from the same tractogram (ex, to compare various
bundle clustering techniques), streamline-wise vote is available to find the
streamlines most often included in the bundle.

Formerly: scil_perform_majority_vote.py
"""


import argparse
import logging

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import get_reference_info
import nibabel as nib
import numpy as np
from scipy.sparse import dok_matrix

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, assert_headers_compatible)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractograms.tractogram_operations import (
    intersection_robust, union_robust)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_bundles', nargs='+',
                   help='Input bundles filename(s). All tractograms must have '
                        'identical headers.')
    p.add_argument('output_prefix',
                   help='Output prefix. Ex: my_path/voting_. The suffixes '
                        'will be: streamlines.trk and voxels.nii.gz')

    p.add_argument('--ratio_voxels', type=float, nargs='?', const=0.5,
                   help='Threshold on the ratio of bundles with at least one '
                        'streamine in a \ngiven voxel to consider it '
                        'as part of the \'gold standard\'. Default if '
                        'set: 0.5.')
    p.add_argument('--ratio_streamlines', type=float, nargs='?', const=0.5,
                   help='If all bundles come from the same tractogram, use '
                        'this to generate \na voting for streamlines too. The '
                        'associated value is the threshold on the ratio of \n'
                        'bundles including the streamline to consider it '
                        'as part of the \'gold standard\'. [0.5]')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bundles, args.reference)
    output_streamlines_filename = '{}streamlines.trk'.format(
        args.output_prefix)
    output_voxels_filename = '{}voxels.nii.gz'.format(args.output_prefix)
    assert_outputs_exist(parser, args, [output_voxels_filename,
                                        output_streamlines_filename])
    assert_headers_compatible(parser, args.in_bundles,
                              reference=args.reference)

    if not 0 <= args.ratio_voxels <= 1 or not 0 <= args.ratio_streamlines <= 1:
        parser.error('Ratios must be between 0 and 1.')

    fusion_streamlines = []
    if not args.reference:
        args.reference = args.in_bundles[0]
    sft_list = []
    for name in args.in_bundles:
        tmp_sft = load_tractogram_with_reference(parser, args, name)
        tmp_sft.to_vox()
        tmp_sft.to_corner()
        sft_list.append(tmp_sft)
        fusion_streamlines.append(tmp_sft.streamlines)

    fusion_streamlines, _ = union_robust(fusion_streamlines)

    transformation, dimensions, _, _ = get_reference_info(args.reference)
    volume = np.zeros(dimensions)
    streamlines_vote = dok_matrix((len(fusion_streamlines),
                                   len(args.in_bundles)))

    for i in range(len(args.in_bundles)):
        sft = sft_list[i]
        binary = compute_tract_counts_map(sft.streamlines, dimensions)
        volume[binary > 0] += 1

        if args.ratio_streamlines is not None:
            _, indices = intersection_robust([fusion_streamlines,
                                              sft.streamlines])
            streamlines_vote[list(indices), [i]] += 1

    if args.ratio_streamlines is not None:
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
