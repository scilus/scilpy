#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import argparse
from itertools import chain
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import random
import sys
from operator import itemgetter
import json
import logging
import copy

from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel
from dipy.tracking.streamline import select_random_set_of_streamlines, set_number_of_points
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
import numpy as np
from numpy import corrcoef

from scilpy.io.utils import (add_overwrite_arg, add_reference_arg, add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_tractogram_with_reference)

from scilpy.tractograms.tractogram_operations import perform_tractogram_operation_on_lines, intersection
from scipy.stats import entropy

DESCRIPTION = """
    SATA style bundle saturation, save a json file of the measure and a plot.
    Without the --whole_brain option, only a approximation of SATA where the bundle is the ground truth.
    In this scenario, your bundle must be oversampled (at least 2x more streamlines than you can possibily imagine.)

    With the --whole_brain option, the sampling simulates generated multiple whole brain generation. Your whole brain
    must have at least 2x what you think is enough for your situation.

    This script assume a very good bundle reconstruction (clean, no outlier)
    """
    # TODO explain linspace vs geomspace
    # TODO how to pick stepsize vs nb_steps


def build_args_parser():

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Substract (remove) streamlines from a file.')

    p.add_argument('bundle',
                   help='The bundle to analyse')

    p.add_argument('output_prefix',
                   help='Prefix for plot and text file')

    p.add_argument('--whole_brain',
                   help='Original whole brain tractogram, allows for a real SATA '
                   'sampling (recommanded).')

    p2 = p.add_mutually_exclusive_group()
    p2.add_argument('--nb_steps', type=int, default=20,
                   help='Number of steps for iteration [%(default)s].')
    p2.add_argument('--stepping_size', type=int,
                   help='Size of chunk during iteration.'
                   '(e.g 1000 for bundles, 100000 for whole brain)')
    p.add_argument('--geomspace', action='store_true',
                   help='Add a greater resolution at small sample.')

    p.add_argument('--recount', type=int, default=10,
                   help='Number of samples for each step [%(default)s]')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p
# TODO Font size
# TODO Text skip in plot
# TODO warn if 1000 when --whole_brain
# TODO add color legend
# TODO plot picked as INT
# TODO --out_suffix/prefix vs --out_dir
# TODO remove initial zeros
# TODO plot all graphs (no text)

def main():
    parser = build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.bundle],
                        [args.whole_brain, args.reference])
    output_data_filename = '{}_data.json'.format(args.output_prefix)
    output_plot_filename = '{}_plot.png'.format(args.output_prefix)
    assert_outputs_exist(parser, args, [output_data_filename,
                                        output_plot_filename])

    if args.stepping_size and args.geomspace:
        parser.error('Impossible!')

    if args.verbose:
        logging.basicConfig(level='INFO')

    b_sft = load_tractogram_with_reference(parser, args, args.bundle)
    resampled_streamlines = set_number_of_points(b_sft.streamlines, 30)
    b_sft = StatefulTractogram.from_sft(resampled_streamlines, b_sft)
    b_sft.to_vox()
    b_sft.to_corner()

    if args.whole_brain:
        wb_sft = load_tractogram_with_reference(parser, args, args.whole_brain)
        resampled_streamlines = set_number_of_points(wb_sft.streamlines, 30)
        wb_sft = StatefulTractogram.from_sft(resampled_streamlines, wb_sft)
        wb_sft.to_vox()
        wb_sft.to_corner()

        # Computing indices both way for tractogram subsampling
        logging.info('Pre-computing intersection for indices')
        _, wb_indices = perform_tractogram_operation_on_lines(intersection,
                                                      [wb_sft.streamlines,
                                                       b_sft.streamlines],
                                                      precision=1)

    volume_dimensions = b_sft.dimensions
    voxel_volume = np.product(b_sft.voxel_sizes)

    # If the whole brain has above 10M streamlines, let's consider that
    # the bundle are saturated-ish at that point
    logging.info('Generating the full bundle density map')
    total_density = compute_tract_counts_map(b_sft.streamlines,
                                                    volume_dimensions).flatten()
    total_volume = float(np.count_nonzero(total_density) * voxel_volume)

    logging.info('Total volume of {}: {}mm3'.format(args.bundle, total_volume))

    
    if args.whole_brain:
        stopping = len(wb_sft.streamlines)
    else:
        stopping = len(b_sft.streamlines)

    if args.stepping_size:
        args.nb_steps = int(np.ceil(stopping / float(args.stepping_size)))
    else:
        args.nb_steps += 1

    measures_data = {}
    for key in ['volume', 'picked', 'correlation', 'dice', 'entropy', 'slope']:
        measures_data[key] = np.zeros((args.nb_steps-2, args.recount))

    flatten_length = np.product(volume_dimensions)
    last_density_array = np.zeros((flatten_length, args.recount),
                                  dtype=np.uint32)
    curr_density_array = np.zeros((flatten_length, args.recount),
                                  dtype=np.uint32)



    if args.geomspace:
        iteration_pick = np.geomspace(max(stopping/1000, 10), stopping, num=args.nb_steps)[1:-1].astype(int)
    else:
        iteration_pick = np.linspace(0, stopping, num=args.nb_steps)[1:-1].astype(int)

    # The saturation curve is sample using an increment defined by the user
    for bin_id, selection in enumerate(iteration_pick):
        it = 0

        # Recount is necessary in case the sampling is just lucky/bad
        while it < args.recount:
            my_randoms = list(np.random.choice(np.arange(stopping), selection,
                                               replace=False))
            if args.whole_brain:
                _, _, tmp = np.intersect1d(my_randoms,
                                           wb_indices,
                                           return_indices=True)
                if len(tmp) <= 1:
                    it += 1
                    continue
                streamlines = b_sft.streamlines[tmp]
            else:
                streamlines = b_sft.streamlines[my_randoms]

            curr_density_array[:, it] = compute_tract_counts_map(
                streamlines,
                volume_dimensions).flatten()

            # Accumulating data for slope/entropy, first 2 bins are skipped
            if bin_id <= 1:
                it += 1
                continue

            measures_data['volume'][bin_id, it] = np.count_nonzero(
                curr_density_array[:, it]) * voxel_volume
            measures_data['picked'][bin_id, it] = len(streamlines)
            measures_data['correlation'][bin_id, it] = corrcoef(
                curr_density_array[:, it],
                total_density)[1, 0]
            measures_data['dice'][bin_id, it], _ = compute_dice_voxel(
                curr_density_array[:, it],
                total_density)
            measures_data['entropy'][bin_id, it] = entropy(
                curr_density_array[:, it]+1e-3,
                last_density_array[:, it]+1e-3)

            last_bin = max(bin_id-5, 0)
            x_ticks = np.arange(bin_id - last_bin) * (1/args.nb_steps)
            tmp_volume_window = measures_data['volume'][last_bin:bin_id,
                                                        it] / total_volume
            measures_data['slope'][bin_id, it], _ = np.polyfit(x_ticks,
                                                               tmp_volume_window,
                                                               1)
            it += 1

        last_density_array = copy.deepcopy(curr_density_array)

        logging.info('---- {} / {} ----'.format(selection, stopping))


        for key in ['volume', 'picked', 'correlation', 'dice', 'entropy', 'slope']:
            if np.sum(measures_data[key][bin_id, :]) < 1e-3:
                continue
            avg = np.average(measures_data[key][bin_id, :])
            std = np.std(measures_data[key][bin_id, :])
            logging.info('{}: {} +/- {}'.format(key, avg, std))

    # Conversion to list for json
    data = {}
    for key in ['volume', 'picked', 'correlation', 'dice', 'entropy', 'slope']:
        data[key] = measures_data[key].tolist()

    with open(output_data_filename, 'w') as outfile:
        json.dump(data, outfile, indent=2, sort_keys=True)

    # Plotting the current bundle for easy Q/C
    avg = np.average(measures_data['volume'], axis=1)
    std = np.std(measures_data['volume'], axis=1)

    ax = plt.plot(iteration_pick, avg)
    ax = plt.fill_between(iteration_pick, avg - std, avg + std,
                          alpha=0.5)

    # Only for simplicity and plotting (remove 1 dimension)
    spacing = np.max(measures_data['volume']) / 50.0
    color = ['red', 'orange', 'pink', 'blue', 'green']
    measures_data['volume'] = np.average(measures_data['volume'], axis=1)
    for i, key in enumerate(['picked', 'correlation', 'dice', 'entropy', 'slope']):
        measures_data[key] = np.average(measures_data[key], axis=1)

        for j in range(1, len(measures_data['volume']), 2):
            if key == 'picked':
                text = '{:.1f}'.format(measures_data[key][j])
            else:
                text = '{:.3f}'.format(measures_data[key][j])
            x = iteration_pick[j]
            y = measures_data['volume'][j] - (i + 3) * spacing
            plt.text(x, y, text,
                     color=color[i], fontsize=4)

    plt.savefig(output_plot_filename, dpi=1200)


if __name__ == "__main__":
    main()
