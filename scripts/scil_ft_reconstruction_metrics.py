#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given ground-truth fibers and a tractogram obtained through fibertube
tracking, computes metrics about the quality of individual fiber
reconstruction.

VC: "Valid Connection": Contains streamlines that ended in the final
    segment of the fiber in which they have been seeded.
IC: "Invalid Connection": Contains streamlines that ended in the final
    segment of another fiber.
NC: "No Connection": Contains streamlines that have not ended in the final
    segment of any fiber.

A coordinate error is the distance between a streamline coordinate and the
closest point on its corresponding fibertube. The average of all coordinate
errors of a streamline is called the "Mean error" or "me".

Computed metrics:
    - truth_vc
        Connections that are valid at ground-truth resolution.
    - truth_ic
    - truth_nc
    - res_vc
        Connections that are valid at degraded resolution.
    - res_ic
    - res_nc
    - me_min
    - me_max
    - me_mean
        Average mean error
    - me_med
        Median mean error
"""
import os
import argparse
import logging
import numpy as np
import nibabel as nib

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from scilpy.tractanalysis.fibertube_scoring import \
    resolve_origin_seeding, endpoint_connectivity, mean_reconstruction_error
from scilpy.io.utils import (load_dictionary,
                             save_dictionary)
from scilpy.tractograms.streamline_operations import \
    get_streamlines_as_fixed_array
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg,
                             add_bbox_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_centroids',
                   help='Path to the tractogram file containing the \n'
                   'fibertubes\' centroids (must be .trk or .tck).')

    p.add_argument('in_diameters',
                   help='Path to a text file containing a list of the \n'
                   'diameters of each fibertube in mm (.txt). Each line \n'
                   'corresponds to the identically numbered centroid.')

    p.add_argument('in_tractogram',
                   help='Path to a text file containing the ground-truth \n'
                   'reconstruction (must be .trk or .tck). \n')

    p.add_argument('in_seeds',
                   help='Path to a text file containing a list of the \n'
                   'seeds used to propagate each streamline of the \n'
                   'reconstruction.')

    p.add_argument('in_config',
                   help='Path to a text file containing the fibertube \n'
                   'parameters used for the tracking process.')

    p.add_argument('out_metrics',
                   help='Output file containing the computed measures and \n'
                   'metrics (must be .txt).')

    p.add_argument('--single_diameter', action='store_true',
                   help='If set, the first diameter found in \n'
                   '[in_diameters] will be repeated for each fiber.')

    p.add_argument('--save_error_tractogram', action='store_true',
                   help='If set, a .trk file will be saved, containing a \n'
                   'visual representation of the per-coordinate error of \n'
                   'each streamline relative to the fiber they have been \n'
                   'seeded in. The file name is derived from the \n'
                   'out_metrics parameter.')

    p.add_argument('--rng_seed', type=int, default=0,
                   help='If set, all random values will be generated \n'
                   'using the specified seed. [%(default)s]')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_bbox_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('numba').setLevel(logging.WARNING)

    if not nib.streamlines.is_supported(args.in_centroids):
        parser.error('Invalid input streamline file format (must be trk ' +
                     'or tck): {0}'.format(args.in_centroids))

    out_metrics_no_ext, ext = os.path.splitext(args.out_metrics)

    if ext != '.txt':
        parser.error('Invalid output file format (must be txt): {0}'
                     .format(args.out_metrics))

    assert_inputs_exist(parser, [args.in_centroids, args.in_diameters,
                                 args.in_seeds, args.in_config,
                                 args.in_tractogram])
    assert_outputs_exist(parser, args, [args.out_metrics])

    logging.debug('Loading centroid tractogram & diameters')
    truth_sft = load_tractogram_with_reference(parser, args,
                                               args.in_centroids)
    truth_sft.to_voxmm()
    truth_sft.to_center()
    fibers, fibers_length = get_streamlines_as_fixed_array(
        truth_sft.get_streamlines_copy())
    diameters = np.loadtxt(args.in_diameters, dtype=np.float64)
    if args.single_diameter:
        diameter = diameters if np.ndim(diameters) == 0 else diameters[0]
        diameters = np.full(len(fibers), diameter)

    logging.debug('Loading reconstructed tractogram')
    in_sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    in_sft.to_voxmm()
    in_sft.to_center()
    streamlines, streamlines_length = get_streamlines_as_fixed_array(
        in_sft.get_streamlines_copy())

    logging.debug("Loading seeds")
    seeds = np.loadtxt(args.in_seeds)
    if len(seeds.shape) != 2:
        seeds = [seeds]
    seeds_fiber = resolve_origin_seeding(seeds, fibers, diameters)

    logging.debug("Loading config")
    config = load_dictionary(args.in_config)
    step_size = float(config['step_size'])
    sampling_radius = float(config['sampling_radius'])

    if len(seeds_fiber) != len(streamlines):
        raise ValueError('Could not resolve origin seeding regions')
    for num in seeds_fiber:
        if num == -1:
            raise ValueError('Could not resolve origin seeding regions')

    logging.debug("Computing endpoint connectivity")
    rand_gen = np.random.default_rng(args.rng_seed)
    (truth_vc, truth_ic, truth_nc,
     res_vc, res_ic, res_nc) = endpoint_connectivity(
        step_size, sampling_radius,
        fibers, fibers_length,
        diameters, streamlines,
        seeds_fiber, rand_gen)

    logging.debug("Computing reconstruction error")
    (mean_errors,
     error_tractogram) = mean_reconstruction_error(fibers, fibers_length,
                                                   diameters,
                                                   streamlines,
                                                   streamlines_length,
                                                   seeds_fiber,
                                                   args.save_error_tractogram)

    measures = {
        'truth_vc_ratio': len(truth_vc)/len(streamlines),
        'truth_ic_ratio': len(truth_ic)/len(streamlines),
        'truth_nc_ratio': len(truth_nc)/len(streamlines),
        'res_vc_ratio': len(res_vc)/len(streamlines),
        'res_ic_ratio': len(res_ic)/len(streamlines),
        'res_nc_ratio': len(res_nc)/len(streamlines),
        'me_min': np.min(mean_errors),
        'me_max': np.max(mean_errors),
        'me_mean': np.mean(mean_errors),
        'me_med': np.median(mean_errors),
    }
    save_dictionary(measures, args.out_metrics, args.overwrite)

    if args.save_error_tractogram:
        sft = StatefulTractogram.from_sft(error_tractogram, truth_sft)
        save_tractogram(sft, out_metrics_no_ext + '.trk',
                        bbox_valid_check=False)


if __name__ == '__main__':
    main()
