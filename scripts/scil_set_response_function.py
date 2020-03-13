#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Replace the fiber response function in the FRF file.
Use this script when you want to use a fixed response function
and keep the mean b0.

The FRF file is obtained from scil_compute_ssst_frf.py
"""

from __future__ import division, print_function

import argparse
from ast import literal_eval
import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exist, load_frf,
    save_frf)
from scilpy.reconst.frf import get_frf_components


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('frf_file', metavar='input',
                   help='Path of the FRF file.')
    p.add_argument('new_frf', metavar='tuple',
                   help='Replace the response function with\n'
                        'this fiber response function x 10**-4 (e.g. '
                        '15,4,4).')
    p.add_argument('output_frf_file', metavar='output',
                   help='Path of the new FRF file.')
    p.add_argument('--no_factor', action='store_true',
                   help='If supplied, the fiber response function is\n'
                        'evaluated without the x 10**-4 factor. [%(default)s].')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.frf_file)
    assert_outputs_exist(parser, args, args.output_frf_file)

    frf_response = load_frf(args.frf_file)

    new_frf = np.array(literal_eval(args.new_frf), dtype=np.float64)
    if not args.no_factor:
        new_frf *= 10 ** -4
    _, b0_mean = get_frf_components(frf_response)

    response = np.concatenate([new_frf, [b0_mean]])

    save_frf(args.output_frf_file, response)


if __name__ == "__main__":
    main()
