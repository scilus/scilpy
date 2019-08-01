#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import logging
from nibabel.streamlines import load, save, Tractogram
import numpy as np

from scilpy.tracking.tools import resample_streamlines
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exists,
                             add_overwrite_arg, add_verbose_arg)


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Resample a set of streamlines.\n'
                    'WARNING: data_per_point is not carried')
    p.add_argument('in_tractogram',
                   help='Streamlines input file name.')
    p.add_argument('out_tractogram',
                   help='Streamlines output file name.')
    p.add_argument('--npts',
                   default=0, type=int,
                   help='Number of points per streamline in the output. [%(default)s]')
    p.add_argument('--arclength',
                   default=False,
                   help='Whether to downsample using arc length ' +
                   'parametrization. [%(default)s]')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():

    parser = _build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    assert_inputs_exist(parser, [args.in_tractogram])
    assert_outputs_exists(parser, args, args.out_tractogram)

    tractogram_file = load(args.in_tractogram)
    streamlines = list(tractogram_file.streamlines)

    new_streamlines = resample_streamlines(streamlines,
                                           args.npts,
                                           args.arclength)

    new_tractogram = Tractogram(
        new_streamlines,
        data_per_streamline=tractogram_file.tractogram.data_per_streamline,
        affine_to_rasmm=np.eye(4))

    save(new_tractogram, args.out_tractogram, header=tractogram_file.header)


if __name__ == "__main__":
    main()
