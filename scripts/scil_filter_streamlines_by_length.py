#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import logging

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.tracking.tools import filter_streamlines_by_length
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Filter streamlines by length.')
    p.add_argument('in_tractogram',
                   help='Streamlines input file name.')
    p.add_argument('out_tractogram',
                   help='Streamlines output file name.')
    p.add_argument('--minL', default=0., type=float,
                   help='Minimum length of streamlines. [%(default)s]')
    p.add_argument('--maxL', default=np.inf, type=float,
                   help='Maximum length of streamlines. [%(default)s]')
    p.add_argument('--no_empty', action='store_true',
                   help='Do not write file if there is no streamline.')
    p.add_argument('--display_counts', action='store_true',
                   help='Print streamline count before and after filtering')

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    add_json_args(p)

    return p


def main():

    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    new_streamlines, \
        new_per_point, \
        new_per_streamline = filter_streamlines_by_length(sft,
                                                          args.minL,
                                                          args.maxL)

    new_sft = StatefulTractogram.from_sft(new_streamlines, sft,
                                          data_per_streamline=new_per_streamline,
                                          data_per_point=new_per_point)

    if not new_streamlines:
        if args.no_empty:
            logging.debug("The file {} won't be written "
                          "(0 streamline).".format(args.out_tractogram))

            return

        logging.debug('The file {} contains 0 streamline'.format(
            args.out_tractogram))

    save_tractogram(new_sft, args.out_tractogram)

    if args.display_counts:
        tc_bf = len(sft.streamlines)
        tc_af = len(new_streamlines)
        print(json.dumps({'tract_count_before_filtering': int(tc_bf),
                          'tract_count_after_filtering': int(tc_af)},
                         indent=args.indent))


if __name__ == "__main__":
    main()
