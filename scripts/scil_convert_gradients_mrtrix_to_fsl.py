#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert bval/bvec MRtrix style to FSL style.
"""

import argparse
import logging

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_gradients_filenames_valid,
                             assert_outputs_exist)
from scilpy.utils.bvec_bval_tools import mrtrix2fsl


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('mrtrix_enc',
                   help='Path to the gradient directions encoding file. (.b)')
    p.add_argument('fsl_bval',
                   help='Path to output FSL b-value file (.bval).')
    p.add_argument('fsl_bvec',
                   help='Path to output FSL gradient directions file (.bvec).')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_gradients_filenames_valid(parser, args.mrtrix_enc, 'mrtrix')
    assert_gradients_filenames_valid(parser, [args.fsl_bval, args.fsl_bvec],
                                     'fsl')
    assert_outputs_exist(parser, args, [args.fsl_bval, args.fsl_bvec])

    mrtrix2fsl(args.mrtrix_enc, args.fsl_bval, args.fsl_bvec)


if __name__ == "__main__":
    main()
