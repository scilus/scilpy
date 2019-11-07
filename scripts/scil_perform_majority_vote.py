#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible, get_reference_info
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from scipy.sparse import dok_matrix

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.streamlines import (perform_streamlines_operation,
                                      intersection, union)

DESCRIPTION = """
Use multiple bundles to perform a voxel-wise vote (occurence across input).
If streamlines originate from the same tractogram, streamline-wise vote
is available.
Input tractograms have to have identical header (register).
"""


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION)

    p.add_argument('in_bundles', nargs='+',
                   help='Input bundles filename.')

    add_reference(p)

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

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
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
    for name in args.in_bundles:
        fusion_streamlines.extend(
            load_tractogram_with_reference(parser, args, name).streamlines)

    fusion_streamlines, _ = perform_streamlines_operation(union,
                                                          [fusion_streamlines],
                                                          0)
    fusion_streamlines = ArraySequence(fusion_streamlines)
    if args.reference:
        reference_file = args.reference
    else:
        reference_file = args.in_bundles[0]

    transformation, dimensions, _, _ = get_reference_info(reference_file)
    volume = np.zeros(dimensions)
    streamlines_vote = dok_matrix((len(fusion_streamlines),
                                   len(args.in_bundles)))

    for i, name in enumerate(args.in_bundles):
        if not is_header_compatible(reference_file, name):
            raise ValueError('Both headers are not the same')
        sft = load_tractogram_with_reference(parser, args, name)
        bundle = sft.get_streamlines_copy()
        sft.to_vox()
        bundle_vox_space = sft.get_streamlines_copy()
        binary = compute_tract_counts_map(bundle_vox_space, dimensions)
        volume[binary > 0] += 1

        if args.same_tractogram:
            _, indices = perform_streamlines_operation(intersection,
                                                       [fusion_streamlines,
                                                        bundle], 0)
            streamlines_vote[list(indices), i] += 1

    if args.same_tractogram:
        real_indices = []
        for i in range(len(fusion_streamlines)):
            ratio_value = int(args.ratio_streamlines*len(args.in_bundles))
            if np.sum(streamlines_vote[i]) >= ratio_value:
                real_indices.append(i)

        new_streamlines = fusion_streamlines[real_indices]

        sft = StatefulTractogram(new_streamlines, reference_file, Space.RASMM)
        save_tractogram(sft, output_streamlines_filename)

    volume[volume < int(args.ratio_streamlines*len(args.in_bundles))] = 0
    volume[volume > 0] = 1
    nib.save(nib.Nifti1Image(volume.astype(np.uint8), transformation),
             output_voxels_filename)


if __name__ == "__main__":
    main()
