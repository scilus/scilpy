#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper for SyNb0 available in Dipy, to run it on a single subject.
Requires Skull-Strip b0 and t1w images as input, the script will normalize the
t1w's WM to 110, co-register both images, then register it to the appropriate
template, run SyNb0 and then transform the result back to the original space.

SyNb0 is a deep learning model that predicts a synthetic a distortion-free
b0 image from a distorted b0 and T1w.

This script must be used carefully, as it is meant to be used in an
environment with the following dependencies already installed (not installed by
default in Scilpy):
- tensorflow-addons
- tensorrt
- tensorflow
"""

import argparse
import logging

import nibabel as nib

from scilpy.image.volume_b0_synthesis import compute_b0_synthesis
from scilpy.io.image import get_data_as_mask
from scilpy.io.fetcher import get_synb0_template_path
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             assert_headers_compatible)

EPILOG = """
[1] Schilling, Kurt G., et al. "Synthesized b0 for diffusion distortion
  correction (Synb0-DisCo)." Magnetic resonance imaging 64 (2019): 62-70.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EPILOG)
    p.add_argument('in_b0',
                   help='Input b0 image.')
    p.add_argument('in_b0_mask',
                   help='Input b0 mask.')
    p.add_argument('in_t1',
                   help='Input t1w image.')
    p.add_argument('in_t1_mask',
                   help='Input t1w mask.')
    p.add_argument('out_b0',
                   help='Output b0 image without distortion.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, [args.in_b0, args.in_b0_mask,
                                 args.in_t1, args.in_t1_mask])
    assert_outputs_exist(parser, args, args.out_b0)
    assert_headers_compatible(parser, [args.in_b0, args.in_b0_mask])
    assert_headers_compatible(parser, [args.in_t1, args.in_t1_mask])

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    logging.info('The usage of synthetic b0 is not fully tested.'
                 'Be careful when using it.')

    # Loading
    template_img = nib.load(get_synb0_template_path())
    template_data = template_img.get_fdata()

    b0_img = nib.load(args.in_b0)
    b0_data = b0_img.get_fdata()
    b0_mask_data = get_data_as_mask(nib.load(args.in_b0_mask))

    t1_img = nib.load(args.in_t1)
    t1_data = t1_img.get_fdata()
    t1_mask_data = get_data_as_mask(nib.load(args.in_t1_mask))

    b0_bet_data = b0_data * b0_mask_data
    t1_bet_data = t1_data * t1_mask_data

    # Processing
    verbose = True if args.verbose in ['INFO', 'DEBUG'] else False
    rev_b0 = compute_b0_synthesis(t1_data, t1_bet_data, b0_data, b0_bet_data,
                                  template_data, t1_img.affine, b0_img.affine,
                                  template_img.affine, verbose)

    # Saving
    dtype = b0_img.get_data_dtype()
    nib.save(nib.Nifti1Image(rev_b0.astype(dtype), b0_img.affine),
             args.out_b0)


if __name__ == "__main__":
    main()
