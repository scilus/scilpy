#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to normalize gradients bvecs.
"""

import argparse
import logging

import numpy as np

from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg, add_verbose_arg)
from scilpy.gradients.bvec_bval_tools import normalize_bvecs
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_bvec',
                   help='Path to the gradient file (.bvec).')

    p.add_argument('out_bvec',
                   help='Output bvec file.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bvec)
    assert_outputs_exist(parser, args, args.out_bvec)

    bvecs = np.loadtxt(args.in_bvec)
    np.savetxt(args.out_bvec, normalize_bvecs(bvecs))


if __name__ == "__main__":
    main()
