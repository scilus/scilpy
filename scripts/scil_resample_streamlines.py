#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging

from nibabel.streamlines import load, save, Tractogram
import numpy as np

from scilpy.tracking.tools import resample_streamlines
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exists,
                             add_overwrite_arg)


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Subsample a set of streamlines.\n'
                    'WARNING: data_per_point is not carried')
    p.add_argument(
        'input', action='store',  metavar='input',
        type=str,  help='Streamlines input file name.')
    p.add_argument(
        'output', action='store',  metavar='output',
        type=str,  help='Streamlines output file name.')
    p.add_argument(
        '--npts', dest='npts', action='store', metavar=' ',
        default=0, type=int,
        help='Number of points per streamline in the output. [%(default)s]')
    p.add_argument(
        '--arclength', dest='arclength', action='store_true', default=False,
        help='Whether to downsample using arc length parametrization. ' +
             '[%(default)s]')

    p.add_argument('-v', action='store_true', dest='verbose',
                   help='Produce verbose output. [%(default)s]')

    add_overwrite_arg(p)

    return p


def main():

    parser = _build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.input])
    assert_outputs_exists(parser, args, args.output)

    tractogramFile = load(args.input)
    streamlines = list(tractogramFile.streamlines)

    new_streamlines = resample_streamlines(streamlines,
                                           args.npts,
                                           args.arclength)

    new_tractogram = Tractogram(
        new_streamlines,
        data_per_streamline=tractogramFile.tractogram.data_per_streamline,
        affine_to_rasmm=np.eye(4))

    save(new_tractogram, args.output, header=tractogramFile.header)


if __name__ == "__main__":
    main()
