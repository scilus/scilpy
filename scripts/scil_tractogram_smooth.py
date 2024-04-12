#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script will smooth the streamlines, usually to remove the 'wiggles' in
probabilistic tracking.

Two choices of methods are available:
- Gaussian will use the surrounding coordinates for smoothing. Streamlines are
resampled to 1mm step-size and the smoothing is performed on the coordinate
array. The sigma will be indicative of the  number of points surrounding the
center points to be used for blurring.
- Spline will fit a spline curve to every streamline using a sigma and the
number of control points. The sigma represents the allowed distance from the
control points. The final streamlines are obtained by evaluating the spline at
constant intervals so that it will have the same number of points as initially.

This script enforces endpoints to remain the same.

WARNING:
- too low of a sigma (e.g: 1) with a lot of control points (e.g: 15)
will create crazy streamlines that could end up out of the bounding box.
- data_per_point will be lost.

Formerly: scil_smooth_streamlines.py
"""

import argparse
import logging

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamlinespeed import compress_streamlines

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist, add_compression_arg)
from scilpy.tractograms.streamline_operations import smooth_line_gaussian, \
    smooth_line_spline


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Input tractography file.')

    p.add_argument('out_tractogram',
                   help='Output tractography file.')

    sub_p = p.add_mutually_exclusive_group(required=True)
    sub_p.add_argument('--gaussian', metavar='SIGMA', type=int,
                       help='Sigma for smoothing. Use the value of surronding'
                            '\nX,Y,Z points on the streamline to blur the '
                            'streamlines.\nA good sigma choice would be '
                            'around 5.')
    sub_p.add_argument('--spline', nargs=2, metavar=('SIGMA', 'NB_CTRL_POINT'),
                       type=int,
                       help='Sigma for smoothing. Model each streamline as a '
                            'spline.\nA good sigma choice would be around 5 '
                            'and control point around 10.')

    add_compression_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    smoothed_streamlines = []
    for streamline in sft.streamlines:
        if args.gaussian:
            tmp_streamlines = smooth_line_gaussian(streamline, args.gaussian)
        else:
            tmp_streamlines = smooth_line_spline(streamline, args.spline[0],
                                                 args.spline[1])

        if args.compress_th:
            tmp_streamlines = compress_streamlines(tmp_streamlines,
                                                   args.compress_th)
        smoothed_streamlines.append(tmp_streamlines)

    smoothed_sft = StatefulTractogram.from_sft(
                        smoothed_streamlines, sft,
                        data_per_streamline=sft.data_per_streamline)
    save_tractogram(smoothed_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
