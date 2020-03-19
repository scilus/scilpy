#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.utils.bvec_bval_tools import fsl2mrtrix

DESCRIPTION = "Script to convert bval/bvec FSL style to MRtrix style."


def _build_args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)

    p.add_argument('fsl_bval',
                   help='path to FSL b-value file.')

    p.add_argument('fsl_bvec',
                   help='path to FSL gradient directions file.')

    p.add_argument('mrtrix_enc',
                   help='path to gradient directions encoding file.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.fsl_bval, args.fsl_bvec])
    assert_outputs_exist(parser, args, [args.mrtrix_enc])

    fsl2mrtrix(args.fsl_bval, args.fsl_bvec, args.mrtrix_enc)


if __name__ == "__main__":
    main()
