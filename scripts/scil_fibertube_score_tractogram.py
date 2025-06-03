#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given ground-truth fibertubes and a tractogram obtained through fibertube
tracking, computes metrics about the quality of individual fiber
reconstruction.

Each streamline is associated with an "Termination fibertube segment", which is
the closest fibertube segment to its before-last coordinate. We then define
the following terms:

VC: "Valid Connection": A streamline whose termination fibertube segment is
the final segment of the fibertube in which is was originally seeded.

IC: "Invalid Connection": A streamline whose termination fibertube segment is
the start or final segment of a fibertube in which is was not seeded.

NC: "No Connection": A streamline whose termination fibertube segment is
not the start or final segment of any fibertube.

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
    - docs/source/documentation/fibertube_tracking.rst
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
    (make_streamlines_forward_only,
     associate_seeds_to_fibertubes,
     endpoint_connectivity,
     mean_reconstruction_error)
from scilpy.tractograms.streamline_operations import \
    get_streamlines_as_fixed_array
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg,
                             add_json_args)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_fibertubes',
                   help='Path to the tractogram (must be .trk) file \n'
                        'containing fibertubes. They must have their \n'
                        'respective diameter saved as data_per_streamline.')

    p.add_argument('in_tracking',
                   help='Path to the tractogram file (must be .trk) \n'
                   'containing the reconstruction of ground-truth \n'
                   'fibertubes made from fibertube tracking. Seeds \n'
                   'used for tracking must be saved as \n'
                   'data_per_streamline.')

    p.add_argument('in_config',
                   help='Path to a json file containing the fibertube \n'
                   'parameters used for the tracking process.')

    p.add_argument('out_metrics',
                   help='Output file containing the computed measures and \n'
                   'metrics (must be .json).')

    p.add_argument('--save_error_tractogram', action='store_true',
                   help='If set, a .trk file will be saved, containing a \n'
                   'visual representation of all the coordinate absolute \n'
                   'errors of the entire tractogram. The file name is \n'
                   'derived from the out_metrics parameter.')

    p.add_argument(
        '--out_tracked_fibertubes', type=str, default=None,
        help='If set, the fibertubes that were used for seeding will be \n'
        'saved separately at the specified location (must be .trk or \n'
        '.tck). This parameter is not required for scoring the tracking \n'
        'result, as the seeding information of each streamline is always \n'
        'saved as data_per_streamline.')

    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    logging.getLogger('numba').setLevel(logging.WARNING)

    if os.path.splitext(args.in_fibertubes)[1] != '.trk':
        parser.error('Invalid input streamline file format (must be trk):' +
                     '{0}'.format(args.in_fibertubes))

    if not nib.streamlines.is_supported(args.in_tracking):
        parser.error('Invalid output streamline file format (must be trk ' +
                     'or tck): {0}'.format(args.in_tracking))

    if os.path.splitext(args.in_config)[1] != '.json':
        parser.error('Invalid input streamline file format (must be json):' +
                     '{0}'.format(args.in_config))

    out_metrics_no_ext, ext = os.path.splitext(args.out_metrics)
    if ext != '.json':
        parser.error('Invalid output file format (must be json): {0}'
                     .format(args.out_metrics))

    if args.out_tracked_fibertubes:
        if not nib.streamlines.is_supported(args.out_tracked_fibertubes):
            parser.error('Invalid output streamline file format (must be ' +
                         'trk or tck):' +
                         '{0}'.format(args.out_tracked_fibertubes))

    assert_inputs_exist(parser, [args.in_fibertubes, args.in_config,
                                 args.in_tracking])
    assert_outputs_exist(parser, args, [args.out_metrics],
                         [args.out_tracked_fibertubes])

    our_space = Space.VOXMM
    our_origin = Origin('center')

    logging.debug('Loading centerline tractogram & diameters')
    truth_sft = load_tractogram(args.in_fibertubes, 'same',
                                to_space=our_space,
                                to_origin=our_origin, bbox_valid_check=False)
    centerlines = truth_sft.get_streamlines_copy()
    centerlines, centerlines_length = get_streamlines_as_fixed_array(
        centerlines)

    if "diameters" not in truth_sft.data_per_streamline:
        parser.error('No diameters found as data per streamline in ' +
                     args.in_fibertubes)
    diameters = np.reshape(truth_sft.data_per_streamline['diameters'],
                           len(centerlines))

    logging.debug('Loading reconstructed tractogram')
    in_sft = load_tractogram(args.in_tracking, 'same',
                             to_space=our_space,
                             to_origin=our_origin, bbox_valid_check=False)
    seeds = in_sft.data_per_streamline['seeds']
    seed_ids = np.ravel(in_sft.data_per_streamline['seed_ids']).astype(int)
    streamlines = in_sft.get_streamlines_copy()
    streamlines = make_streamlines_forward_only(streamlines, seed_ids)
    streamlines, streamlines_length = get_streamlines_as_fixed_array(
        streamlines)

    logging.debug("Loading seeds")
    if "seeds" not in in_sft.data_per_streamline:
        parser.error('No seeds found as data per streamline in ' +
                     args.in_tracking)

    seeded_fibertube_indices = associate_seeds_to_fibertubes(
        seeds, centerlines, diameters)

    logging.debug("Loading config")
    with open(args.in_config, 'r') as f:
        config = json.load(f)
    blur_radius = float(config['blur_radius'])

    if len(seeded_fibertube_indices) != len(streamlines):
        raise ValueError('Could not resolve origin seeding regions')
    for num in seeded_fibertube_indices:
        if num == -1:
            raise ValueError('Could not resolve origin seeding regions')

    if args.out_tracked_fibertubes:
        # Set for removing doubles
        tracked_fibertubes_indices = set(seeded_fibertube_indices)
        tracked_fibertubes = []

        for fi in tracked_fibertubes_indices:
            tracked_fibertubes.append(centerlines[fi][:centerlines_length[fi]])

        tracked_sft = StatefulTractogram.from_sft(tracked_fibertubes,
                                                  truth_sft)
        save_tractogram(tracked_sft, args.out_tracked_fibertubes,
                        bbox_valid_check=False)

    logging.debug("Computing endpoint connectivity")
    vc, ic, nc, endpoint_distances = endpoint_connectivity(
        blur_radius, centerlines, centerlines_length, diameters, streamlines,
        streamlines_length, seeded_fibertube_indices)

    logging.debug("Computing reconstruction error")
    mean_errors, error_tractogram = mean_reconstruction_error(
        centerlines, centerlines_length, diameters, streamlines,
        streamlines_length, seeded_fibertube_indices,
        args.save_error_tractogram)

    metrics = {
        'vc_ratio': len(vc)/len(seeds),
        'ic_ratio': len(ic)/len(seeds),
        'nc_ratio': len(nc)/len(seeds),
        'mae_min': np.min(mean_errors),
        'mae_max': np.max(mean_errors),
        'mae_mean': np.mean(mean_errors),
        'mae_med': np.median(mean_errors),
        'endpoint_dist_min': np.min(endpoint_distances),
        'endpoint_dist_max': np.max(endpoint_distances),
        'endpoint_dist_mean': np.mean(endpoint_distances),
        'endpoint_dist_med': np.median(endpoint_distances)
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
