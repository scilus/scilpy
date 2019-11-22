#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from nibabel.streamlines import load, save, Tractogram
import numpy as np

from scilpy.tracking.tools import filter_streamlines_by_length
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Filter streamlines by length.')
    p.add_argument('in_tractogram', type=str,
                   help='Streamlines input file name.')
    p.add_argument('out_tractogram', type=str,
                   help='Streamlines output file name.')
    p.add_argument('--minL', default=0., type=float,
                   help='Minimum length of streamlines. [%(default)s]')
    p.add_argument('--maxL', default=np.inf, type=float,
                   help='Maximum length of streamlines. [%(default)s]')

    add_overwrite_arg(p)

    return p


def main():

    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    tractogram_file = load(args.in_tractogram)
    streamlines = list(tractogram_file.streamlines)

    data_per_point = tractogram_file.tractogram.data_per_point
    data_per_streamline = tractogram_file.tractogram.data_per_streamline

    new_streamlines, new_per_point, new_per_streamline = filter_streamlines_by_length(
                                                             streamlines,
                                                             data_per_point,
                                                             data_per_streamline,
                                                             args.minL,
                                                             args.maxL)

    new_tractogram = Tractogram(new_streamlines,
                                data_per_streamline=new_per_streamline,
                                data_per_point=new_per_point,
                                affine_to_rasmm=np.eye(4))

    save(new_tractogram, args.out_tractogram, header=tractogram_file.header)


if __name__ == "__main__":
    main()
