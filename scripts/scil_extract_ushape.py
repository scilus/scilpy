#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be used to extract U-Shape streamlines.
The main idea comes from trackvis code:

pt 1: 1 st end point
pt 2: 1/3 location on the track
pt 3: 2/3 location on the track
pt 4: 2nd end point

Compute 3 normalized vectors:
v1: pt1 -> pt2
v2: pt2 -> pt3
v3: pt3 -> pt4

U-factor:dot product of  v1 X v2 and v2 X v3.
X is the cross product of two vectors.
----------------------------------------------------------------------------
"""

import argparse
import json
import logging

from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             check_tracts_same_format)
from scilpy.utils.streamlines import filter_tractogram_data
from scilpy.tractanalysis.features import detect_ushape


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('in_tractogram',
                   help='Tractogram input file name.')
    p.add_argument('out_tractogram',
                   help='Output tractogram with ushape.')
    p.add_argument('--ufactor', default=[0.5, 1], type=float, nargs=2,
                   help='U factor defines U-shapeness of the track.')
    p.add_argument('--remaining_tractogram',
                   help='If set, saves remaining streamlines.')
    p.add_argument('--display_counts', action='store_true',
                   help='Print streamline count before and after filtering')

    add_overwrite_arg(p)
    add_reference_arg(p)
    add_json_args(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram,
                         optional=args.remaining_tractogram)
    check_tracts_same_format(parser, [args.in_tractogram, args.out_tractogram,
                                      args.remaining_tractogram])

    #if args.threshold <= 0:
    #    parser.error('Threshold "{}" '.format(args.ufactor) +
    #                 'must be greater than 0')

    tractogram = load_tractogram_with_reference(
        parser, args, args.in_tractogram)

    streamlines = tractogram.streamlines

    ids_c = []

    ids_l = []

    if len(streamlines) > 1:
        ids_c = detect_ushape(streamlines, args.ufactor)
        ids_l = np.setdiff1d(np.arange(len(streamlines)), ids_c)
    else:
        parser.error(
            'Zero or one streamline in {}'.format(args.in_tractogram) +
            '. The file must have more than one streamline.')

    if len(ids_c) > 0:
        sft_c = filter_tractogram_data(tractogram, ids_c)
        save_tractogram(sft_c, args.out_tractogram)
    else:
        logging.warning(
            'No clean streamlines in {}'.format(args.in_tractogram))

    if args.display_counts:
        sc_bf = len(tractogram.streamlines)
        sc_af = len(sft_c.streamlines)
        print(json.dumps({'streamline_count_before_filtering': int(sc_bf),
                         'streamline_count_after_filtering': int(sc_af)},
                         indent=args.indent))

    if len(ids_l) == 0:
        logging.warning('No loops in {}'.format(args.in_tractogram))
    elif args.remaining_tractogram:
        sft_l = filter_tractogram_data(tractogram, ids_l)
        save_tractogram(sft_l, args.remaining_tractogram)


if __name__ == "__main__":
    main()
