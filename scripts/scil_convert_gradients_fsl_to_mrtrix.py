#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert bval/bvec FSL style to MRtrix style.
"""

import argparse
import logging

from scilpy.io.utils import (assert_gradients_filenames_valid,
                             assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg, add_verbose_arg)
from scilpy.utils.bvec_bval_tools import fsl2mrtrix


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('fsl_bval',
                   help='Path to FSL b-value file (.bval).')

    p.add_argument('fsl_bvec',
                   help='Path to FSL gradient directions file (.bvec).')

    p.add_argument('mrtrix_enc',
                   help='Path to gradient directions encoding file (.b).')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_gradients_filenames_valid(parser, [args.fsl_bval, args.fsl_bvec],
                                     'fsl')
    assert_gradients_filenames_valid(parser, args.mrtrix_enc, 'mrtrix')
    assert_inputs_exist(parser, [args.fsl_bval, args.fsl_bvec])
    assert_outputs_exist(parser, args, args.mrtrix_enc)

    fsl2mrtrix(args.fsl_bval, args.fsl_bvec, args.mrtrix_enc)


if __name__ == "__main__":
    main()
