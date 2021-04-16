#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute signal to noise ratio (SNR) in a region of interest (ROI)
of a DWI volume.

It will compute the SNR for all DWI volumes of the input image seperately.
The output will contain the SNR
The mean of the signal is computed inside the mask.
The standard deviation of the noise is estimated inside noise_mask.
If it's not supplied, it will be estimated using the data outside medotsu.

If verbose is True, the SNR for every DWI volume will be outputed.

This works best in a well-defined ROI such as the corpus callosum.
It is heavily dependent on the ROI and its quality.
"""

import argparse

from scilpy.io.utils import (add_verbose_arg, assert_inputs_exist)
from scilpy.utils.image import compute_snr


def _build_arg_parser():

    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_dwi',
                   help='Path of the input diffusion volume.')

    p.add_argument('in_bval',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('in_bvec',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('in_roi', action='store', metavar='mask_roi',
                   help='Binary mask of the region used to estimate SNR.')

    p.add_argument('--noise', action='store', dest='noise_mask',
                   metavar='noise_mask',
                   help='Binary mask used to estimate the noise.')

    p.add_argument('--b0_thr', type=float, default=0.0,
                   help='All b-values with values less than or equal '
                        'to b0_thr are considered as b0s i.e. without '
                        'diffusion weighting. [Default: 0.0]')

    p.add_argument('-out_basename',
                   help='Path and prefix for the various saved file.')

    add_verbose_arg(p)

    return p


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval,
                                 args.in_bvec, args.in_mask])

    compute_snr(args.in_dwi, args.in_bval, args.in_bvec, args.b0_thr,
                args.mask_roi,
                noise_mask=args.noise_mask,
                basename=args.basename,
                verbose=args.verbose)


if __name__ == "__main__":
    main()
