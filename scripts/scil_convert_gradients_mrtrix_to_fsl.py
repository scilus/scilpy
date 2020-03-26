#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to convert bval/bvec MRtrix style to FSL style.
"""

import argparse

from scilpy.io.utils import (assert_gradients_filenames_valid,
                             assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.utils.bvec_bval_tools import mrtrix2fsl


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('mrtrix_enc',
                   help='Gradient directions encoding file. (.b)')
    p.add_argument('fsl_bval',
                   help='path to output FSL b-value file.')
    p.add_argument('fsl_bvec',
                   help='path to output FSL gradient directions file.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_gradients_filenames_valid(parser, args.mrtrix_enc, 'mrtrix')
    assert_gradients_filenames_valid(parser, [args.fsl_bval, args.fsl_bvec],
                                     'fsl')
    assert_outputs_exist(parser, args, [args.fsl_bval, args.fsl_bvec])

    mrtrix2fsl(args.mrtrix_enc, args.fsl_bval, args.fsl_bvec)


if __name__ == "__main__":
    main()
