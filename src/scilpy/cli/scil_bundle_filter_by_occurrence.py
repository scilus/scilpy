#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use multiple versions of a same bundle and detect the most probable voxels by
using a threshold on the occurrence, voxel-wise. With threshold 0.5, this is
a majority vote. This is useful to generate an average representation from
bundles of a given population.

If streamlines originate from the same tractogram (ex, to compare various
bundle clustering techniques), streamline-wise vote is available to find the
streamlines most often included in the bundle (use --ratio_streamlines).
"""

import argparse
import logging

from dipy.io.utils import get_reference_info
import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, assert_headers_compatible,
                             ranged_type)

from scilpy.tractanalysis.multi_bundle_operations import filter_by_occurrence
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_bundles', nargs='+',
                   help='Input bundles filename(s). All tractograms must have '
                        'identical headers.')
    p.add_argument('output_prefix',
                   help='Output prefix. Ex: my_path/voting_. The suffixes '
                        'will be: voxels.nii.gz and, if --ratio_streamlines '
                        'is used, streamlines.trk.')

    p.add_argument('--ratio_voxels',
                   type=ranged_type(float, 0, 1), nargs='?', const=0.5,
                   help='Threshold on the ratio of bundles with at least one '
                        'streamine in a \ngiven voxel to consider it '
                        'as part of the \'gold standard\'. Default if '
                        'set: 0.5.')
    p.add_argument('--ratio_streamlines',
                   type=ranged_type(float, 0, 1), nargs='?', const=0.5,
                   help='If all bundles come from the same tractogram, use '
                        'this to generate \na voting for streamlines too. The '
                        'associated value is the threshold on the ratio of \n'
                        'bundles including the streamline to consider it '
                        'as part of the \'gold standard\'. Default if set: '
                        '[0.5]')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Checks
    assert_inputs_exist(parser, args.in_bundles, args.reference)
    output_streamlines_filename = '{}streamlines.trk'.format(
        args.output_prefix)
    output_voxels_filename = '{}voxels.nii.gz'.format(args.output_prefix)
    assert_outputs_exist(parser, args, [output_voxels_filename,
                                        output_streamlines_filename])
    assert_headers_compatible(parser, args.in_bundles,
                              reference=args.reference)

    # Load and merge bundles
    if not args.reference:
        args.reference = args.in_bundles[0]
    sft_list = []
    for name in args.in_bundles:
        tmp_sft = load_tractogram_with_reference(parser, args, name)
        tmp_sft.to_vox()
        tmp_sft.to_corner()
        sft_list.append(tmp_sft)

    affine, vol_dim, _, _ = get_reference_info(args.reference)

    # Processing
    volume, _ = filter_by_occurrence(
        sft_list, vol_dim, ratio_voxels=args.ratio_voxels,
        ratio_streamlines=args.ratio_streamlines)
    nib.save(nib.Nifti1Image(volume.astype(np.uint8), affine),
             output_voxels_filename)


if __name__ == "__main__":
    main()
