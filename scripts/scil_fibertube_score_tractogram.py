#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given ground-truth fibertubes and a tractogram obtained through fibertube
tracking, computes metrics about the quality of individual fiber
reconstruction.

VC: "Valid Connection": A streamline that passes WITHIN the final segment of
    the fibertube in which it was seeded.
IC: "Invalid Connection": A streamline that ended in the final segment of
    another fibertube.
NC: "No Connection": A streamlines that has not ended in the final segment
    of any fibertube.

Res_VC: "Resolution-wise Valid Connection": A streamline that passes closer
    than [blur_darius] away from the last segment of the fibertube in which it
    was seeded.
Res_IC: "Resolution-wise Invalid Connection": A streamline that passes closer
    than [blur_darius] away from the first or last segment of another
    fibertube.
Res_NC: "Resolution-wise No Connection": A streamlines that does not pass
    closer than [blur_radius] away from the first or last segment of any
    fibertube.

The "absolute error" of a coordinate is the distance in mm between that
coordinate and the closest point on its corresponding fibertube. The average
of all coordinate absolute errors of a streamline is called the "Mean absolute
error" or "mae".

Computed metrics:
    - vc_ratio
        Number of VC divided by the number of streamlines.
    - ic_ratio
        Number of IC divided by the number of streamlines.
    - nc_ratio
        Number of NC divided by the number of streamlines.
    - res_vc_ratio
        Number of Res_VC divided by the number of streamlines.
    - res_ic_ratio
        Number of Res_IC divided by the number of streamlines.
    - res_nc_ratio
        Number of Res_NC divided by the number of streamlines.
    - mae_min
        Minimum MAE for the tractogram.
    - mae_max
        Maximum MAE for the tractogram.
    - mae_mean
        Average MAE for the tractogram.
    - mae_med
        Median MAE for the tractogram.

See also:
    - scil_tractogram_filter_collisions.py to prepare data for fibertube
      tracking
    - scil_fibertube_tracking.py to perform a fibertube tracking
    - docs/fibertube/DEMO.md
"""

import os
import json
import argparse
import logging
import numpy as np
import nibabel as nib

from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import save_tractogram, load_tractogram
from scilpy.tractanalysis.fibertube_scoring import \
    resolve_origin_seeding, endpoint_connectivity, mean_reconstruction_error
from scilpy.tractograms.streamline_operations import \
    get_streamlines_as_fixed_array
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg,
                             add_json_args)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_fibertubes',
                   help='Path to the tractogram file (must be .trk) \n'
                   'containing ground-truth fibertubes. They must be: \n'
                   '1- Void of any collision. \n'
                   '2- With their respective diameter saved \n'
                   'as data_per_streamline. \n'
                   'For both of these requirements, see \n'
                   'scil_tractogram_filter_collisions.')

    p.add_argument('in_tracking',
                   help='Path to the tractogram file (must be .trk) \n'
                   'containing the reconstruction of ground-truth \n'
                   'fibertubes made from fibertube tracking. Seeds \n'
                   'used for tracking must be saved as \n'
                   'data_per_streamline.')

    p.add_argument('in_config',
                   help='Path to a text file containing the fibertube \n'
                   'parameters used for the tracking process.')

    p.add_argument('out_metrics',
                   help='Output file containing the computed measures and \n'
                   'metrics (must be .txt).')

    p.add_argument('--save_error_tractogram', action='store_true',
                   help='If set, a .trk file will be saved, containing a \n'
                   'visual representation of all the coordinate absolute \n'
                   'errors of the entire tractogram. The file name is \n'
                   'derived from the out_metrics parameter.')
    
    p.add_argument(
        '--out_tracked_fibertubes', type=str, default=None,
        help='If set, the fibertubes that were used for seeding will be \n'
        'saved separately at the specified location (must be .trk). This \n'
        'parameter is not required for scoring the tracking result, as the \n'
        'seeding information of each streamline is always saved as \n'
        'data_per_streamline.')

    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    logging.getLogger('numba').setLevel(logging.WARNING)

    if not nib.streamlines.is_supported(args.in_fibertubes):
        parser.error('Invalid input streamline file format (must be trk ' +
                     'or tck): {0}'.format(args.in_fibertubes))

    out_metrics_no_ext, ext = os.path.splitext(args.out_metrics)
    if ext != '.txt':
        parser.error('Invalid output file format (must be txt): {0}'
                     .format(args.out_metrics))

    if args.out_tracked_fibertubes:
        _, out_tracked_fibertubes_ext = os.path.splitext(
            args.out_tracked_fibertubes)
        if out_tracked_fibertubes_ext != '.trk':
            parser.error('Invalid output file format (must be trk): {0}'
                         .format(args.out_tracked_fibertubes))

    assert_inputs_exist(parser, [args.in_fibertubes, args.in_config,
                                 args.in_tracking])
    assert_outputs_exist(parser, args, [args.out_metrics])

    our_space = Space.VOXMM
    our_origin = Origin('center')

    logging.debug('Loading centerline tractogram & diameters')
    truth_sft = load_tractogram(args.in_fibertubes, 'same', our_space,
                                our_origin)
    centerlines = truth_sft.get_streamlines_copy()
    centerlines, centerlines_length = get_streamlines_as_fixed_array(
        centerlines)

    if "diameters" not in truth_sft.data_per_streamline:
        parser.error('No diameters found as data per streamline on ' +
                     args.in_fibertubes)
    diameters = np.reshape(truth_sft.data_per_streamline['diameters'],
                           len(centerlines))

    logging.debug('Loading reconstructed tractogram')
    in_sft = load_tractogram(args.in_tracking, 'same', our_space,
                             our_origin)
    streamlines = in_sft.get_streamlines_copy()
    streamlines, streamlines_length = get_streamlines_as_fixed_array(
        streamlines)

    logging.debug("Loading seeds")
    if "seeds" not in in_sft.data_per_streamline:
        parser.error('No seeds found as data per streamline on ' +
                     args.in_tracking)

    seeds = in_sft.data_per_streamline['seeds']
    seeds_fiber = resolve_origin_seeding(seeds, centerlines, diameters)

    logging.debug("Loading config")
    with open(args.in_config, 'r') as f:
        config = json.load(f)
    step_size = float(config['step_size'])
    blur_radius = float(config['blur_radius'])

    if len(seeds_fiber) != len(streamlines):
        raise ValueError('Could not resolve origin seeding regions')
    for num in seeds_fiber:
        if num == -1:
            raise ValueError('Could not resolve origin seeding regions')

    if args.out_tracked_fibertubes:
        # Set for removing doubles
        tracked_fibertubes_indices = set(seeds_fiber)
        tracked_fibertubes = []

        for fi in tracked_fibertubes_indices:
            tracked_fibertubes.append(centerlines[fi][:centerlines_length[fi]])

        tracked_sft = StatefulTractogram.from_sft(tracked_fibertubes, truth_sft)
        save_tractogram(tracked_sft, args.out_tracked_fibertubes,
                        bbox_valid_check=False)

    logging.debug("Computing endpoint connectivity")
    (truth_vc, truth_ic, truth_nc,
     res_vc, res_ic, res_nc) = endpoint_connectivity(
        step_size, blur_radius,
        centerlines, centerlines_length,
        diameters, streamlines,
        seeds_fiber)

    logging.debug("Computing reconstruction error")
    (mean_errors, error_tractogram) = mean_reconstruction_error(
        centerlines, centerlines_length, diameters, streamlines,
        streamlines_length, seeds_fiber, args.save_error_tractogram)

    metrics = {
        'truth_vc_ratio': len(truth_vc)/len(streamlines),
        'truth_ic_ratio': len(truth_ic)/len(streamlines),
        'truth_nc_ratio': len(truth_nc)/len(streamlines),
        'res_vc_ratio': len(res_vc)/len(streamlines),
        'res_ic_ratio': len(res_ic)/len(streamlines),
        'res_nc_ratio': len(res_nc)/len(streamlines),
        'mae_min': np.min(mean_errors),
        'mae_max': np.max(mean_errors),
        'mae_mean': np.mean(mean_errors),
        'mae_med': np.median(mean_errors),
    }
    with open(args.out_metrics, 'w') as outfile:
        json.dump(metrics, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)

    if args.save_error_tractogram:
        sft = StatefulTractogram.from_sft(error_tractogram, truth_sft)
        save_tractogram(sft, out_metrics_no_ext + '.trk',
                        bbox_valid_check=False)


if __name__ == '__main__':
    main()
