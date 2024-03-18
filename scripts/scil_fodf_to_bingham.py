#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for fitting a Bingham distribution to each fODF lobe, as described
in [1].

The Bingham fit is saved, with each Bingham distribution described by 7
coefficients (for example, for a maximum number of lobes of 5, the number
of coefficients is 7 x 5 = 35 -- less than the number of coefficients for
SH of maximum order 8).

Using 12 threads, the execution takes approximately 30 minutes for a brain with
1mm isotropic resolution.

Formerly: scil_fit_bingham_to_fodf.py
"""

import nibabel as nib
import time
import argparse
import logging

from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, validate_nbr_processes,
                             assert_headers_compatible)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.bingham import (bingham_fit_sh)


EPILOG = """
[1] T. W. Riffert, J. Schreiber, A. Anwander, and T. R. Knösche, “Beyond
    fractional anisotropy: Extraction of bundle-specific structural metrics
    from crossing fiber models,” NeuroImage, vol. 100, pp. 176-191, Oct. 2014,
    doi: 10.1016/j.neuroimage.2014.06.015.

[2] J. Schreiber, T. Riffert, A. Anwander, and T. R. Knösche, “Plausibility
    Tracking: A method to evaluate anatomical connectivity and microstructural
    properties along fiber pathways,” NeuroImage, vol. 90, pp. 163-178, Apr.
    2014, doi: 10.1016/j.neuroimage.2014.01.002.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=EPILOG)
    p.add_argument('in_sh',
                   help='Input SH image.')

    p.add_argument('out_bingham',
                   help='Output Bingham functions image.')

    p.add_argument('--max_lobes', type=int, default=5,
                   help='Maximum number of lobes per voxel'
                        ' to extract. [%(default)s]')
    p.add_argument('--at', type=float, default=0.0,
                   help='Absolute threshold for peaks'
                        ' extraction. [%(default)s]')
    p.add_argument('--rt', type=float, default=0.1,
                   help='Relative threshold for peaks'
                        ' extraction. [%(default)s]')
    p.add_argument('--min_sep_angle', type=float, default=25.,
                   help='Minimum separation angle between'
                        ' two peaks. [%(default)s]')
    p.add_argument('--max_fit_angle', type=float, default=15.,
                   help='Maximum distance in degrees around a peak direction'
                        ' for fitting the Bingham function. [%(default)s]')
    p.add_argument('--mask',
                   help='Optional mask file. Only SH inside'
                        ' the mask are fitted.')

    add_verbose_arg(p)
    add_processes_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_sh, args.mask)
    assert_outputs_exist(parser, args, args.out_bingham)
    assert_headers_compatible(parser, args.in_sh, args.mask)

    sh_im = nib.load(args.in_sh)
    data = sh_im.get_fdata()
    mask = get_data_as_mask(nib.load(args.mask),
                            dtype=bool) if args.mask else None

    # validate number of processes
    nbr_processes = validate_nbr_processes(parser, args)
    logging.info('Number of processes: {}'.format(nbr_processes))

    t0 = time.perf_counter()
    logging.info('Fitting Bingham functions.')
    bingham = bingham_fit_sh(data, args.max_lobes,
                             abs_th=args.at, rel_th=args.rt,
                             min_sep_angle=args.min_sep_angle,
                             max_fit_angle=args.max_fit_angle,
                             mask=mask,
                             nbr_processes=nbr_processes)
    t1 = time.perf_counter()
    logging.info('Fitting done in (s): {0}'.format(t1 - t0))
    nib.save(nib.Nifti1Image(bingham, sh_im.affine), args.out_bingham)


if __name__ == '__main__':
    main()
