#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replace the fiber response function in the FRF file.
Use this script when you want to use a fixed response function
and keep the mean b0.

The FRF file is obtained from scil_frf_ssst.py or scil_frf_msmt.py in the case
of multi-shell data.

Formerly: scil_set_response_function.py
"""

import argparse
import logging
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             add_verbose_arg,
                             assert_outputs_exist)
from scilpy.reconst.frf import replace_frf


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('frf_file', metavar='input',
                   help='Path of the FRF file.')
    p.add_argument('new_frf',
                   help='New response function given as a tuple. We will '
                        'replace the \nresponse function in frf_file with '
                        'this fiber response \nfunction x 10**-4 (e.g. '
                        '15,4,4). \nIf multi-shell, write the first shell,'
                        'then the second shell, \nand the third, etc. '
                        '(e.g. 15,4,4,13,5,5,12,5,5).')
    p.add_argument('output_frf_file', metavar='output',
                   help='Path of the new FRF file.')
    p.add_argument('--no_factor', action='store_true',
                   help='If supplied, the fiber response function is\n'
                        'evaluated without the x 10**-4 factor. [%(default)s].'
                   )

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.frf_file)
    assert_outputs_exist(parser, args, args.output_frf_file)

    frf_file = np.loadtxt(args.frf_file)
    response = replace_frf(frf_file, args.new_frf, args.no_factor)
    np.savetxt(args.output_frf_file, response)


if __name__ == "__main__":
    main()
