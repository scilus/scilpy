#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge a list of Spherical Harmonics files.

This merges the coefficients of multiple Spherical Harmonics files by taking,
for each coefficient, the one with the largest magnitude.

Can be used to merge fODFs computed from different shells into 1, while
conserving the most relevant information.

Based on [1] and [2].

---------------------------------------------------------------------------------
References:
[1] Garyfallidis, E., Zucchelli, M., Houde, J-C., Descoteaux, M.
    How to perform best ODF reconstruction from the Human Connectome
    Project sampling scheme?
    ISMRM 2014.

[2] Khachaturian, M. H., Wisco, J. J., & Tuch, D. S. (2007). Boosting the
    sampling efficiency of q‐ball imaging using multiple wavevector fusion.
    Magnetic Resonance in Medicine: An Official Journal of the International
    Society for Magnetic Resonance in Medicine, 57(2), 289-296.
---------------------------------------------------------------------------------
"""

import argparse
import logging

import numpy as np

from scilpy.io.stateful_image import StatefulImage
from scilpy.io.utils import (add_overwrite_arg, assert_headers_compatible,
                             assert_inputs_exist, assert_outputs_exist,
                             add_verbose_arg)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_shs', nargs="+",
                   help='List of SH files.')
    p.add_argument('out_sh',
                   help='output SH file.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_shs)
    assert_outputs_exist(parser, args, args.out_sh)
    assert_headers_compatible(parser, args.in_shs)

    first_simg = StatefulImage.load(args.in_shs[0], is_orientation=True)
    out_coeffs = first_simg.get_fdata(dtype=np.float32)

    for sh_file in args.in_shs[1:]:
        curr_simg = StatefulImage.load(sh_file, is_orientation=True)
        curr_simg.reorient(first_simg.axcodes)
        im_dat = curr_simg.get_fdata(dtype=np.float32)

        out_coeffs = np.where(np.abs(im_dat) > np.abs(out_coeffs),
                              im_dat, out_coeffs)

    StatefulImage.create_from(
        out_coeffs.astype(np.float32), first_simg,
        is_orientation=True).save(args.out_sh)


if __name__ == '__main__':
    main()
