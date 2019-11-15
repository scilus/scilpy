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

    p.add_argument('mrtrix_enc', type=str,
                   help='Gradient directions encoding file. (.b)')

    p.add_argument('fsl_basename', type=str,
                   help='Output basename gradient directions encoding file. '
                   '(without extension)')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.mrtrix_enc)
    assert_outputs_exist(parser, args, [args.fsl_basename + '.bvec',
                                        args.fsl_basename + '.bval'])

    mrtrix2fsl(args.mrtrix_enc, args.fsl_basename)


if __name__ == "__main__":
    main()
