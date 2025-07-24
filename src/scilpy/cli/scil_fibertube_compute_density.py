#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimates the per-voxel volumetric density of a set of fibertubes. In other
words, how much space is occupied by fibertubes and how much is emptiness.

1. Segments voxels that contain at least a single fibertube.
2. Valid voxels are finely sampled and we count the number of samples that
landed within a fibertube. For each voxel, this number is then divided by
its total amount of samples.
3. By doing the same steps for samples that landed within 2 or more
fibertubes, we can create a density map of the fibertube collisions.

To form fibertubes from a set of streamlines, you can use the scripts:
- scil_tractogram_filter_collisions.py to assign a diameter to each streamline
  and remove all colliding fibertubes.
- scil_tractogram_dps_math.py to assign a diameter without filtering.

See also:
    - docs/source/documentation/fibertube_tracking.rst
"""

import os
import json
import nibabel as nib
import argparse
import logging
import numpy as np

from scilpy.io.streamlines import load_tractogram
from scilpy.tractanalysis.fibertube_scoring import fibertube_density
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg,
                             add_json_args)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('in_fibertubes',
                   help='Path to the tractogram (must be .trk) file \n'
                        'containing fibertubes. They must have their \n'
                        'respective diameter saved as data_per_streamline.')

    p.add_argument('--out_density_map', default=None, type=str,
                   help='Path of the density Nifti image.')

    p.add_argument('--out_density_measures', default=None, type=str,
                   help='Path of the output file containing central \n'
                        'tendency measures about volumetric density. \n'
                        '(Must be .json)')

    p.add_argument('--out_collision_map', default=None, type=str,
                   help='Path of the collision Nifti image.')

    p.add_argument('--out_collision_measures', default=None, type=str,
                   help='Path of the output file containing central \n'
                        'tendency measures about collision density. \n'
                        '(Must be .json)')

    p.add_argument('--samples_per_voxel_axis', default=10, type=int,
                   help='Number of samples to be created along a single \n'
                   'axis of a voxel. The total number of samples in the \n'
                   'voxel will be this number cubed. [%(default)s]')

    add_overwrite_arg(p)
    add_verbose_arg(p)
    add_json_args(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    logging.getLogger('numba').setLevel(logging.WARNING)

    if not (args.out_density_map or args.out_density_measures or
            args.out_collision_map or args.out_collision_measures):
        parser.error('No output argument given')

    if os.path.splitext(args.in_fibertubes)[1] != '.trk':
        parser.error('Invalid input streamline file format (must be trk):' +
                     '{0}'.format(args.in_fibertubes))

    if args.out_density_measures:
        if os.path.splitext(args.out_density_measures)[1] != '.json':
            parser.error('Invalid output file format (must be json): {0}'
                         .format(args.out_density_measures))

    if args.out_collision_measures:
        if os.path.splitext(args.out_collision_measures)[1] != '.json':
            parser.error('Invalid output file format (must be json): {0}'
                         .format(args.out_collision_measures))

    assert_inputs_exist(parser, args.in_fibertubes)
    assert_outputs_exist(parser, args, [],
                         [args.out_density_map, args.out_density_measures,
                          args.out_collision_map, args.out_collision_measures])

    logging.debug('Loading tractogram & diameters')
    sft = load_tractogram(args.in_fibertubes, 'same')
    sft.to_voxmm()
    sft.to_center()

    if "diameters" not in sft.data_per_streamline:
        parser.error('No diameters found as data per streamline in ' +
                     args.in_fibertubes)

    logging.debug('Computing fibertube density')
    (density_grid,
     density_flat,
     collision_grid,
     collision_flat) = fibertube_density(sft, args.samples_per_voxel_axis,
                                         args.verbose != 'INFO')

    logging.debug('Saving output')
    header = nib.Nifti1Header()
    extra = {
        'affine': sft.affine,
        'dimensions': sft.dimensions,
        'voxel_size': sft.voxel_sizes[0],
        'voxel_order': "RAS"
    }

    if args.out_density_map:
        density_img = nib.Nifti1Image(density_grid, sft.affine, header, extra)
        nib.save(density_img, args.out_density_map)

    if args.out_density_measures:
        density_measures = {
            'mean': np.mean(density_flat),
            'median': np.median(density_flat),
            'max': np.max(density_flat),
            'min': np.min(density_flat),
        }
        with open(args.out_density_measures, 'w') as file:
            json.dump(density_measures, file, indent=args.indent,
                      sort_keys=args.sort_keys)

    if args.out_collision_map:
        collision_img = nib.Nifti1Image(collision_grid, sft.affine, header,
                                        extra)
        nib.save(collision_img, args.out_collision_map)

    if args.out_collision_measures:
        collision_measures = {
            'mean': np.mean(collision_flat),
            'median': np.median(collision_flat),
            'max': np.max(collision_flat),
            'min': np.min(collision_flat),
        }
        with open(args.out_collision_measures, 'w') as file:
            json.dump(collision_measures, file, indent=args.indent,
                      sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
