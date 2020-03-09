#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.utils.bvec_bval_tools import mrtrix2fsl

DESCRIPTION = "Script to convert bval/bvec MRtrix style to FSL style."


def _build_args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)

    p.add_argument('mrtrix_enc',
                   help='Gradient directions encoding file. (.b)')
    p.add_argument('fsl_bval',
                   help='path to output FSL b-value file.')
    p.add_argument('fsl_bvec',
                   help='path to output FSL gradient directions file.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.mrtrix_enc)
    assert_outputs_exist(parser, args, [args.fsl_bval, args.fsl_bvec])

    mrtrix2fsl(args.mrtrix_enc, args.fsl_bval, args.fsl_bvec)


if __name__ == "__main__":
    main()
