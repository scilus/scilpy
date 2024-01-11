#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the mean Fiber Response Function from a set of individually
computed Response Functions.

Formerly: scil_compute_mean_frf.py
"""

import argparse
import logging

import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             add_verbose_arg,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('frf_files', metavar='list', nargs='+',
                   help='List of FRF filepaths.')
    p.add_argument('mean_frf', metavar='file',
                   help='Path of the output mean FRF file.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    assert_inputs_exist(parser, args.frf_files)
    assert_outputs_exist(parser, args, args.mean_frf)

    all_frfs = np.zeros((len(args.frf_files), 4))

    for idx, frf_file in enumerate(args.frf_files):
        frf = np.loadtxt(frf_file)

        if not frf.shape[0] == 4:
            raise ValueError('FRF file {} did not contain 4 elements. Invalid '
                             'or deprecated FRF format'.format(frf_file))

        all_frfs[idx] = frf

    final_frf = np.mean(all_frfs, axis=0)

    np.savetxt(args.mean_frf, final_frf)


if __name__ == "__main__":
    main()
