#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compares a gradient table with the Dipy sphere and outputs the maximal angle
between any sphere vertice and its closest bvec.
"""

import argparse
import logging

from dipy.io.gradients import read_bvals_bvecs

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             add_json_args, add_verbose_arg,
                             assert_outputs_exist)
from scilpy.gradients.bvec_bval_tools import compare_bvecs_to_sphere


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_bvec',
                   help='Path to bvec file.')
    p.add_argument('out_metrics',
                   help='Path to a json file in which the validation results\n'
                        'will be saved.')

    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_bvec])
    assert_outputs_exist(parser, args, args.out_metrics)

    logging.debug("Loading bvecs")
    _, bvecs = read_bvals_bvecs(None, args.in_bvec)

    max_angle = compare_bvecs_to_sphere(bvecs)

    print(max_angle)

if __name__ == "__main__":
    main()
