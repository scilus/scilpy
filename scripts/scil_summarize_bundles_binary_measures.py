#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute well-known binary measures between gold standard and bundles.
All tractograms must be trk files and headers must be identical.
The measures can be applied to voxel-wise or streamline-wise representation.

A gold standard must be provided for the desired representation, if only the
streamline-wise representation is provided a voxel-wise gold standard will be
computed. At least one of the two representations is required.
"""

import argparse
import itertools
import logging
import json
import multiprocessing
import os

from dipy.io.streamline import load_tractogram
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_reference_arg,
                             link_bundles_and_reference)
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.reproducibility_measures import binary_classification
from scilpy.utils.streamlines import (perform_streamlines_operation,
                                      intersection)


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('bundles', nargs='+',
                   help='Path of the input bundles')
    p.add_argument('output',
                   help='Path of the output file')
    p.add_argument('--streamlines_measures', nargs=2,
                   metavar=('GOLD STANDARD', 'TRACTOGRAM'),
                   help='The gold standard bundle and the original tractogram')
    p.add_argument('--voxels_measures', nargs=2,
                   metavar=('GOLD STANDARD', 'TRACKING MASK'),
                   help='The gold standard mask and the original tracking mask')
    p.add_argument('--files_exist', action='store_false',
                   help='Disable the verification of input files')
    p.add_argument('--processes', type=int,
                   help='Number of processes to use [ALL]')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def compute_voxel_measures(args):
    bundle_filename, bundle_reference = args[0]
    tracking_mask = args[1]
    gs_binary_3d = args[2]

    if not os.path.isfile(bundle_filename):
        logging.info('{0} does not exist'.format(bundle_filename))
        return None

    bundle_sft = load_tractogram(bundle_filename, bundle_reference)
    bundle_sft.to_vox()
    bundle_sft.to_corner()
    bundle_streamlines = bundle_sft.streamlines
    _, bundle_dimensions, _, _ = bundle_sft.space_attributes

    if not bundle_streamlines:
        logging.info('{0} is empty'.format(bundle_filename))
        return None

    binary_3d = compute_tract_counts_map(bundle_streamlines, bundle_dimensions)
    binary_3d[binary_3d > 0] = 1

    binary_3d_indices = np.where(binary_3d.flatten() > 0)[0]
    gs_binary_3d_indices = np.where(gs_binary_3d.flatten() > 0)[0]

    voxels_binary = binary_classification(binary_3d_indices,
                                          gs_binary_3d_indices,
                                          int(np.prod(tracking_mask.shape)),
                                          mask_count=np.count_nonzero(tracking_mask))

    return dict(zip(['sensitivity_voxels',
                     'specificity_voxels',
                     'precision_voxels',
                     'accuracy_voxels',
                     'dice_voxels',
                     'kappa_voxels',
                     'youden_voxels'],
                    voxels_binary))


def compute_streamlines_measures(args):
    bundle_filename, bundle_reference = args[0]
    wb_streamlines = args[1]
    gs_streamlines_indices = args[2]

    if not os.path.isfile(bundle_filename):
        logging.info('{0} does not exist'.format(bundle_filename))
        return None

    bundle_sft = load_tractogram(bundle_filename, bundle_reference)
    bundle_sft.to_vox()
    bundle_sft.to_corner()
    bundle_streamlines = bundle_sft.streamlines
    _, bundle_dimensions, _, _ = bundle_sft.space_attributes

    if not bundle_streamlines:
        logging.info('{0} is empty'.format(bundle_filename))
        return None

    _, streamlines_indices = perform_streamlines_operation(intersection,
                                                           [wb_streamlines,
                                                               bundle_streamlines],
                                                           precision=0)

    streamlines_binary = binary_classification(streamlines_indices,
                                               gs_streamlines_indices,
                                               len(wb_streamlines))

    return dict(zip(['sensitivity_streamlines',
                     'specificity_streamlines',
                     'precision_streamlines',
                     'accuracy_streamlines',
                     'dice_streamlines',
                     'kappa_streamlines',
                     'youden_streamlines'],
                    streamlines_binary))


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.files_exist:
        assert_inputs_exist(parser, args.bundles)
    assert_outputs_exist(parser, args, [args.output])

    if (not args.streamlines_measures) and (not args.voxels_measures):
        parser.error('At least one of the two modes is needed')

    nbr_cpu = args.processes if args.processes else multiprocessing.cpu_count()
    if nbr_cpu <= 0:
        parser.error('Number of processes cannot be <= 0.')
    elif nbr_cpu > multiprocessing.cpu_count():
        parser.error('Max number of processes is {}. Got {}.'.format(
            multiprocessing.cpu_count(), nbr_cpu))

    all_binary_metrics = []
    bundles_references_tuple_extended = link_bundles_and_reference(
        parser, args, args.bundles)

    if args.streamlines_measures:
        # Gold standard related indices are computed once
        wb_sft = load_tractogram_with_reference(parser, args,
                                                args.streamlines_measures[1])
        wb_sft.to_vox()
        wb_sft.to_corner()
        wb_streamlines = wb_sft.streamlines

        gs_sft = load_tractogram_with_reference(parser, args,
                                                args.streamlines_measures[0])
        gs_sft.to_vox()
        gs_sft.to_corner()
        gs_streamlines = gs_sft.streamlines
        _, gs_dimensions, _, _ = gs_sft.space_attributes

        # Prepare the gold standard only once
        _, gs_streamlines_indices = perform_streamlines_operation(intersection,
                                                                  [wb_streamlines,
                                                                   gs_streamlines],
                                                                  precision=0)

        pool = multiprocessing.Pool(nbr_cpu)
        streamlines_dict = pool.map(compute_streamlines_measures,
                                    zip(bundles_references_tuple_extended,
                                        itertools.repeat(wb_streamlines),
                                        itertools.repeat(gs_streamlines_indices)))
        all_binary_metrics.extend(streamlines_dict)
        pool.close()
        pool.join()

    if not args.voxels_measures:
        gs_binary_3d = compute_tract_counts_map(gs_streamlines,
                                                gs_dimensions)
        gs_binary_3d[gs_binary_3d > 0] = 1

        tracking_mask_data = compute_tract_counts_map(wb_streamlines,
                                                      gs_dimensions)
        tracking_mask_data[tracking_mask_data > 0] = 1
    else:
        gs_binary_3d = nib.load(args.voxels_measures[0]).get_data()
        gs_binary_3d[gs_binary_3d > 0] = 1
        tracking_mask_data = nib.load(args.voxels_measures[1]).get_data()
        tracking_mask_data[tracking_mask_data > 0] = 1

    pool = multiprocessing.Pool(nbr_cpu)
    voxels_binary = pool.map(compute_voxel_measures,
                             zip(bundles_references_tuple_extended,
                                 itertools.repeat(tracking_mask_data),
                                 itertools.repeat(gs_binary_3d)))
    all_binary_metrics.extend(voxels_binary)
    pool.close()
    pool.join()

    # After all processing, write the json file and skip None value
    output_binary_dict = {}
    for binary_dict in all_binary_metrics:
        if binary_dict is not None:
            for measure_name in binary_dict.keys():
                if measure_name not in output_binary_dict:
                    output_binary_dict[measure_name] = []
                output_binary_dict[measure_name].append(
                    binary_dict[measure_name])

    with open(args.output, 'w') as outfile:
        json.dump(output_binary_dict, outfile, indent=1)


if __name__ == "__main__":
    main()
