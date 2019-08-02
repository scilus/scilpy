#! /usr/bin/env python
'''
    Script to convert bval/bvec fsl style to mrtrix style.
'''

import argparse

from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exists,
                             add_overwrite_arg)
from scilpy.utils.bvec_bval_tools import fsl2mrtrix


def _build_args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description='Convert bval/bvec fsl ' +
                                'style  to mrtrix style.')

    p.add_argument('fsl_bval',
                   help='path to fsl b-value file.')

    p.add_argument('fsl_bvec',
                   help='path to fsl gradient directions file.')

    p.add_argument('mrtrix_enc',
                   help='path to gradient directions encoding file.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.fsl_bval, args.fsl_bvec])
    assert_outputs_exists(parser, args, [args.mrtrix_enc])

    fsl2mrtrix(args.fsl_bval, args.fsl_bvec, args.mrtrix_enc)


if __name__ == "__main__":
    main()
