#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from nibabel.streamlines import load, save, Tractogram
import numpy as np

from scilpy.tracking.tools import resample_streamlines
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Resample a set of streamlines.\n'
                    'WARNING: data_per_point is not carried')
    p.add_argument('in_tractogram',
                   help='Streamlines input file name.')
    p.add_argument('nb_pts_per_streamline', type=int,
                   help='Number of points per streamline in the output.')
    p.add_argument('out_tractogram',
                   help='Streamlines output file name.')
    p.add_argument('--arclength', action="store_true",
                   help='Whether to downsample using arc length ' +
                   'parametrization. [%(default)s]')

    add_overwrite_arg(p)

    return p


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    tractogram_file = load(args.in_tractogram)
    streamlines = list(tractogram_file.streamlines)

    new_streamlines = resample_streamlines(streamlines,
                                           args.nb_pts_per_streamline,
                                           args.arclength)

    new_tractogram = Tractogram(
        new_streamlines,
        data_per_streamline=tractogram_file.tractogram.data_per_streamline,
        affine_to_rasmm=np.eye(4))

    save(new_tractogram, args.out_tractogram, header=tractogram_file.header)


if __name__ == "__main__":
    main()
