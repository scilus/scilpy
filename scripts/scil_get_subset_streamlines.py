#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from nibabel.streamlines import load, save, Tractogram
import numpy as np

from scilpy.tracking.tools import get_subset_streamlines
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exists,
                             add_overwrite_arg)


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Get a subset of streamlines.')
    p.add_argument('in_tractogram',
                   help='Streamlines input file name.')
    p.add_argument('max_num_streamlines',
                   default=0, type=int,
                   help='Maximum number of streamlines to output. [all]')
    p.add_argument('out_tractogram',
                   help='Streamlines output file name.')
    p.add_argument('--seed',
                   default=None, type=int,
                   help='Use a specific random seed for the resampling.')

    add_overwrite_arg(p)

    return p


def main():

    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram])
    assert_outputs_exists(parser, args, args.out_tractogram)

    tractogram_file = load(args.in_tractogram)
    streamlines = list(tractogram_file.streamlines)

    data_per_point = tractogram_file.tractogram.data_per_point
    data_per_streamline = tractogram_file.tractogram.data_per_streamline

    new_streamlines, new_per_point, new_per_streamline = get_subset_streamlines(
                                                       streamlines,
                                                       data_per_point,
                                                       data_per_streamline,
                                                       args.max_num_streamlines,
                                                       args.seed)

    new_tractogram = Tractogram(new_streamlines,
                                data_per_point=new_per_point,
                                data_per_streamline=new_per_streamline,
                                affine_to_rasmm=np.eye(4))

    save(new_tractogram, args.out_tractogram, header=tractogram_file.header)


if __name__ == "__main__":
    main()
