#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script performs a bundle saturation analysis (SATA) on a given tractogram,
either as a standalone bundle or within the context of a whole-brain tractogram.

It simulates a sampling of your bundle/tractogram and shows how the metrics
evolve with the number of streamlines picked. This allows you to determine
the number of streamlines needed to get a stable values.

Input:
- A tractogram file representing a bundle of interest (`bundle` argument).
- Optionally, a whole-brain tractogram for comprehensive analysis
    (`--whole_brain` argument).
Both of these (if used) should be extremely dense tractograms, to ensure
more than enough streamlines are available for sampling.

Output:
- A JSON file containing metrics such as volume, dice coefficient, entropy,
    etc., computed at different sampling levels.
- A set of PNG plots for each metric, illustrating how the metric values
    change with different sampling levels.
"""

import argparse
import copy
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             add_json_args,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             load_tractogram_with_reference)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel
from scilpy.tractograms.tractogram_operations import (intersection,
                                                      perform_tractogram_operation_on_lines)


def build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('bundle',
                   help='Path to the bundle file to be analyzed. Must be in a '
                        'supported format (.trk, .tck, etc.).')

    p.add_argument('--out_dir', default='',
                   help='Directory where all output files will be saved. '
                        '\nIf not specified, outputs will be saved in the current '
                        'directory.')
    p.add_argument('--out_prefix', default='',
                   help='Prefix for output files. Useful for distinguishing between '
                        'different runs.')

    p.add_argument('--whole_brain',
                   help='Path to the whole-brain tractogram file.'
                        '\nProvides a more realistic SATA (Saturation Analysis) sampling.'
                        '\nHighly recommended for comprehensive analysis.')

    p2 = p.add_mutually_exclusive_group()
    p2.add_argument('--nb_steps', type=int, default=20,
                    help='Number of steps to perform in the iteration.'
                         '\nDefines how many different sizes of the tractogram or '
                         'bundle will be analyzed [%(default)s].')
    p2.add_argument('--stepping_size', type=int,
                    help='Size of the step for each iteration. '
                         'For bundles, smaller sizes like 1000 are recommended.'
                         '\nwhile for whole-brain tractograms, larger sizes like '
                         '100000 may be more appropriate.')

    p.add_argument('--geomspace', action='store_true',
                   help='Use geometric spacing for the steps. '
                        'This provides higher resolution at smaller sizes.')
    p.add_argument('--nb_recount', type=int, default=10,
                   help='Number of resamples at each step to average out random '
                        'sampling variations. Default is 10.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_json_args(p)
    add_overwrite_arg(p)

    return p


def compute_measures(b_sft, nb_recount, nb_steps, stepping_size,
                     geomspace=False, wb_sft=None):
    """
    Compute various metrics for the given bundle and optionally the whole brain
    tractogram.

    Args:
        b_sft (StatefulTractogram): The bundle tractogram.
        nb_recount (int): Number of samples for each step.
        nb_steps (int): Total number of steps.
        stepping_size (int): Size of each step.
        geomspace (bool, optional): Use geometric spacing. Default is False.
        wb_sft (StatefulTractogram, optional): whole-brain tractogram.
                                               Default is None.

    Returns:
        dict, np.array: The computed metrics and iteration picks.
    """
    b_sft.to_vox()
    b_sft.to_corner()

    volume_dimensions = b_sft.dimensions
    voxel_volume = np.product(b_sft.voxel_sizes)

    logging.info('Generating the full bundle density map')
    total_density = compute_tract_counts_map(b_sft.streamlines,
                                             volume_dimensions).flatten()
    total_volume = float(np.count_nonzero(total_density) * voxel_volume)

    logging.info('Total volume of {}mm3'.format(total_volume))

    if wb_sft:
        wb_sft.to_vox()
        wb_sft.to_corner()

        # Computing indices both way for tractogram subsampling
        logging.info('Pre-computing intersection for indices')
        _, wb_indices = perform_tractogram_operation_on_lines(intersection,
                                                              [wb_sft.streamlines,
                                                               b_sft.streamlines],
                                                              precision=1)
        stopping = len(wb_sft.streamlines)
    else:
        stopping = len(b_sft.streamlines)

    if stepping_size:
        nb_steps = int(np.ceil(stopping / float(stepping_size)))
    else:
        nb_steps += 1

    # Initializing the data structure
    measures_data = {}
    for key in ['volume', 'picked', 'w_dice', 'dice', 'entropy', 'slope']:
        measures_data[key] = np.zeros((nb_steps-1, nb_recount))

    flatten_length = np.product(volume_dimensions)

    # Either sample using a geometric spacing or a linear one
    if geomspace:
        iteration_pick = np.geomspace(max(stopping/1000, 10),
                                      stopping, num=nb_steps)[1:].astype(int)
    else:
        iteration_pick = np.linspace(0, stopping, num=nb_steps)[1:].astype(int)

    last_density = np.zeros((flatten_length,), dtype=np.uint32)
    curr_density = np.zeros((flatten_length,), dtype=np.uint32)

    for bin_id, selection in enumerate(iteration_pick):
        # Useless to pick all streamlines, so pick 2/3 of the last bin
        if bin_id == len(iteration_pick) - 1:
            selection = (2*iteration_pick[-1] + iteration_pick[-2]) // 3

        # nb_recount is necessary in case the sampling is just lucky/bad
        for r in range(nb_recount):
            my_randoms = list(np.random.choice(np.arange(stopping), selection,
                                               replace=False))
            if wb_sft:
                _, _, tmp = np.intersect1d(my_randoms,
                                           wb_indices,
                                           return_indices=True)
                if len(tmp) <= 1:
                    continue
                streamlines = b_sft.streamlines[tmp]
            else:
                streamlines = b_sft.streamlines[my_randoms]

            curr_density = compute_tract_counts_map(streamlines,
                                                    volume_dimensions).flatten()

            dice, w_dice = compute_dice_voxel(curr_density, total_density)

            measures_data['volume'][bin_id, r] = np.count_nonzero(
                curr_density) * voxel_volume
            measures_data['picked'][bin_id, r] = len(streamlines)
            measures_data['dice'][bin_id, r] = dice
            measures_data['w_dice'][bin_id, r] = w_dice

            # Accumulating data for slope/entropy, first 2 bins are skipped
            if bin_id <= 1:
                continue
            measures_data['entropy'][bin_id, r] = entropy(curr_density+1e-3,
                                                          last_density+1e-3)

            last_bin_id = max(bin_id-5, 0)
            x_ticks = np.arange(bin_id - last_bin_id) * (1/nb_steps)
            win_vol = measures_data['volume'][last_bin_id:bin_id,
                                              r] / total_volume
            measures_data['slope'][bin_id, r], _ = np.polyfit(x_ticks,
                                                              win_vol, 1)

        last_density = copy.deepcopy(curr_density)

        # Logging for user readability
        logging.info('\n---- {} / {} ----'.format(selection, stopping))
        for key in ['volume', 'picked', 'w_dice', 'dice', 'entropy', 'slope']:
            if np.sum(measures_data[key][bin_id, :]) < 1e-3:
                continue
            avg = np.average(measures_data[key][bin_id, :])
            std = np.std(measures_data[key][bin_id, :])
            logging.info('{}: {} +/- {}'.format(key, avg, std))

    return measures_data, iteration_pick


def plot_measures(measures_data, iteration_pick, out_prefix, out_dir):
    """
    Generate plots for each metric computed.

    Args:
        measures_data (dict): Dictionary containing the computed metrics.
        iteration_pick (np.array): Array containing the iteration picks.
        out_prefix (str): Prefix for the output files.
        out_dir (str): Output directory.
    """
    # Remove zero values for plotting
    zeros = np.sum(measures_data['slope'], axis=1) < 1e-3
    for measure in measures_data.keys():
        measures_data[measure] = measures_data[measure][~zeros, :]
    iteration_pick = iteration_pick[~zeros]

    # Iterate over each measure to create separate graphs
    for measure in measures_data.keys():
        plt.figure()

        avg = np.average(measures_data[measure], axis=1)
        std = np.std(measures_data[measure], axis=1)

        # Scatter plot for individual data points
        for i, x in enumerate(iteration_pick):
            y = measures_data[measure][i, :]
            plt.scatter([x] * len(y), y, color='blue', alpha=0.5, s=4)

        # Line plot for average values and text for readability
        plt.plot(iteration_pick, avg, color='red', label='Average')

        plt.fill_between(iteration_pick, avg - std, avg + std,
                         color='red', alpha=0.25)

        for i, (x, y, s) in enumerate(zip(iteration_pick, avg, std)):
            if i % 2 == 0:
                plt.text(x*1.02, y*1.02,
                         f"{y:.3f} +/- {s:.3f}", fontsize=4, ha='right')

        # Set y-axis limits for specific measures
        if measure in ['dice', 'w_dice', 'entropy']:
            plt.ylim(0, 1.05)

        plt.title(f"Measure: {measure}")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.legend()

        # Save the plot
        filename = os.path.join(out_dir, "{}_{}.png".format(out_prefix,
                                                            measure))
        plt.savefig(filename, dpi=1200)
        plt.close()


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.bundle],
                        [args.whole_brain, args.reference])
    if args.out_prefix and args.out_prefix[-1] == '_':
        args.out_prefix = args.out_prefix[:-1]
    output_data_filename = os.path.join(args.out_dir,
                                        '{}_data.json'.format(args.out_prefix))
    output_plot_filename = os.path.join(args.out_dir,
                                        '{}_plot.png'.format(args.out_prefix))
    assert_output_dirs_exist_and_empty(parser, args, [], optional=args.out_dir)
    assert_outputs_exist(parser, args, [output_data_filename,
                                        output_plot_filename])

    if args.stepping_size and args.geomspace:
        parser.error('Impossible!')

    if args.verbose:
        logging.basicConfig(level='INFO')

    b_sft = load_tractogram_with_reference(parser, args, args.bundle)
    if args.whole_brain:
        if args.stepping_size and args.stepping_size == 1000:
            logging.warning('You are using a stepping size of 1000, '
                            'this is probably too small for a whole brain.')
        wb_sft = load_tractogram_with_reference(parser, args, args.whole_brain)
    else:
        wb_sft = None

    measures_data, iteration_pick = compute_measures(b_sft, args.nb_recount,
                                                     args.nb_steps,
                                                     args.stepping_size,
                                                     args.geomspace, wb_sft)
    # Conversion to list for json
    data = {}
    for key in ['volume', 'picked', 'w_dice', 'dice', 'entropy', 'slope']:
        data[key] = measures_data[key].tolist()

    with open(output_data_filename, 'w') as outfile:
        json.dump(data, outfile, sort_keys=args.sort_keys, indent=args.indent)

    plot_measures(measures_data, iteration_pick, args.out_prefix, args.out_dir)


if __name__ == "__main__":
    main()
