#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prints information on a loaded tractogram: number of streamlines, and
mean / min / max / std of
    - length in number of points
    - length in mm
    - step size.

For trk files: also prints the data_per_point and data_per_streamline keys.

See also:
    - scil_header_print_info.py to see the header, affine, volume dimension.
    - scil_bundle_shape_measures.py to see bundle-specific information.
"""

import argparse
import json
import logging

from dipy.tracking.streamlinespeed import length
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Tractogram file.')
    add_reference_arg(p)
    add_verbose_arg(p)
    add_json_args(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram, args.reference)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    lengths = [len(s) for s in sft.streamlines]
    lengths_mm = list(length(sft.streamlines))

    sft.to_voxmm()
    steps = [np.sqrt(np.sum(np.diff(s, axis=0) ** 2, axis=1))
             for s in sft.streamlines]
    steps = np.hstack(steps)

    print(json.dumps(
        {'min_length_mm': float(np.min(lengths_mm)),
         'mean_length_mm': float(np.mean(lengths_mm)),
         'max_length_mm': float(np.max(lengths_mm)),
         'std_length_mm': float(np.std(lengths_mm)),
         'min_length_nb_points': float(np.min(lengths)),
         'mean_length_nb_points': float(np.mean(lengths)),
         'max_length_nb_points': float(np.max(lengths)),
         'std_length_nb_points': float(np.std(lengths)),
         'min_step_size': float(np.min(steps)),
         'mean_step_size': float(np.mean(steps)),
         'max_step_size': float(np.max(steps)),
         'std_step_size': float(np.std(steps)),
         'data_per_point_keys': list(sft.data_per_point.keys()),
         'data_per_streamline_keys': list(sft.data_per_streamline.keys())
         },
        indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
