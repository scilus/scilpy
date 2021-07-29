#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Dumb script to resample a tractogram to a set number of streamlines.
Might be useful to build training sets for machine learning algorithms, to
upsample under-represented bundles or downsample over-represented bundles.

Works by either selecting a subset of streamlines or by generating new
streamlines by adding gaussian noise to existing ones.

It is recommended to use "scil_smooth_streamlines" afterwards to clean the
noisy new streamlines.
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.streamlines import (downsample_tractogram,
                                      upsample_tractogram)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Input tractography file.')
    p.add_argument('out_tractogram',
                   help='Output tractography file.')
    p.add_argument('nb_streamlines', type=int,
                   help='Number of streamlines to resample the tractogram to.')
    p.add_argument('--std', type=float, default=0.1,
                   help='Noise to add to existing streamlines to generate ' +
                        ' new ones')

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    original_number = len(sft.streamlines)

    if args.nb_streamlines > original_number:
        sft = upsample_tractogram(sft, args.nb_streamlines, args.std)
    elif args.nb_streamlines < original_number:
        sft = downsample_tractogram(sft, args.nb_streamlines)
    # else do nothing

    save_tractogram(sft, args.out_tractogram)


if __name__ == "__main__":
    main()
