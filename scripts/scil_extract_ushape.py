#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script extracts streamlines depending on their U-shapeness.
This script is a replica of Trackvis method.

When ufactor is close to:
*  0 it defines straight streamlines
*  1 it defines U-fibers
* -1 it defines S-fibers
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
from scilpy.tractanalysis.features import detect_ushape


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('in_tractogram',
                   help='Tractogram input file name.')
    p.add_argument('out_tractogram',
                   help='Output tractogram file name.')
    p.add_argument('--minU',
                   default=0.5, type=float,
                   help='Min ufactor value. [%(default)s]')
    p.add_argument('--maxU',
                   default=1.0, type=float,
                   help='Max ufactor value. [%(default)s]')

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

    if not(-1 <= args.minU <= 1 and -1 <= args.maxU <= 1):
        parser.error('Min-Max ufactor "{},{}" '.format(args.minU, args.maxU) +
                     'must be between -1 and 1.')

    sft = load_tractogram_with_reference(
        parser, args, args.in_tractogram)

    ids_c = []
    ids_l = []

    if len(sft.streamlines) > 1:
        ids_c = detect_ushape(sft, args.minU, args.maxU)
        ids_l = np.setdiff1d(np.arange(len(sft.streamlines)), ids_c)
    else:
        parser.error(
            'Zero or one streamline in {}'.format(args.in_tractogram) +
            '. The file must have more than one streamline.')

    if len(ids_c) > 0:
        save_tractogram(sft[ids_c], args.out_tractogram)
    else:
        logging.warning(
            'No u-shape streamlines in {}'.format(args.in_tractogram))

    if args.display_counts:
        sc_bf = len(sft.streamlines)
        sc_af = len(ids_c)
        print(json.dumps({'streamline_count_before_filtering': int(sc_bf),
                         'streamline_count_after_filtering': int(sc_af)},
                         indent=args.indent))

    if len(ids_l) == 0:
        logging.warning('No remaining streamlines '
                        'in {}'.format(args.remaining_tractogram))
    elif args.remaining_tractogram:
        save_tractogram(sft[ids_l], args.remaining_tractogram)


if __name__ == "__main__":
    main()
