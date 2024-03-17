#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute fODF lobe-specific metrics derived from a Bingham
distribution fit, as described in [1]. Resulting metrics are fiber density
(FD), fiber spread (FS) and fiber fraction (FF) [2].

The Bingham coefficients volume comes from scil_fodf_to_bingham.py.

A lobe's FD is the integral of the Bingham function on the sphere. It
represents the density of fibers going through a given voxel for a given
fODF lobe (fixel). A lobe's FS is the ratio of its FD on its maximum AFD. It
is at its minimum for a sharp lobe and at its maximum for a wide lobe. A lobe's
FF is the ratio of its FD on the total FD in the voxel.

Using 12 threads, the execution takes 10 minutes for FD estimation for a brain
with 1mm isotropic resolution. Other metrics take less than a second.

Formerly: scil_compute_lobe_specific_fodf_metrics.py
"""

import nibabel as nib
import time
import argparse
import logging

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, validate_nbr_processes,
                             assert_headers_compatible)
from scilpy.reconst.bingham import (compute_fiber_density,
                                    compute_fiber_spread,
                                    compute_fiber_fraction)


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
    p.add_argument('in_bingham',
                   help='Input Bingham nifti image.')

    p.add_argument('--out_fd',
                   help='Path to output fiber density. [fd.nii.gz]')
    p.add_argument('--out_fs',
                   help='Path to output fiber spread. [fs.nii.gz]')
    p.add_argument('--out_ff',
                   help='Path to fiber fraction file. [ff.nii.gz]')
    p.add_argument('--not_all', action='store_true',
                   help='Do not compute all metrics. Then, please provide '
                        'the output paths of the files you need.')
    p.add_argument('--mask',
                   help='Optional mask image. Only voxels inside '
                        'the mask are computed.')

    p.add_argument('--nbr_integration_steps', type=int, default=50,
                   help='Number of integration steps along the theta axis for'
                        ' fiber density estimation. [%(default)s]')

    add_verbose_arg(p)
    add_processes_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if not args.not_all:
        args.out_fd = args.out_fd or 'fd.nii.gz'
        args.out_fs = args.out_fs or 'fs.nii.gz'
        args.out_ff = args.out_ff or 'ff.nii.gz'

    arglist = [args.out_fd, args.out_fs, args.out_ff]
    if args.not_all and not any(arglist):
        parser.error('At least one output file must be specified.')

    outputs = [args.out_fd, args.out_fs, args.out_ff]
    assert_inputs_exist(parser, args.in_bingham, args.mask)
    assert_outputs_exist(parser, args, [], optional=outputs)
    assert_headers_compatible(parser, args.in_bingham, args.mask)

    bingham_im = nib.load(args.in_bingham)
    bingham = bingham_im.get_fdata()
    mask = get_data_as_mask(nib.load(args.mask),
                            dtype=bool) if args.mask else None

    nbr_processes = validate_nbr_processes(parser, args)

    t0 = time.perf_counter()
    logging.info('Computing fiber density.')
    fd = compute_fiber_density(bingham, m=args.nbr_integration_steps,
                               mask=mask, nbr_processes=nbr_processes)
    t1 = time.perf_counter()
    logging.info('FD computed in (s): {0}'.format(t1 - t0))
    if args.out_fd:
        nib.save(nib.Nifti1Image(fd, bingham_im.affine), args.out_fd)

    if args.out_fs:
        t0 = time.perf_counter()
        logging.info('Computing fiber spread.')
        fs = compute_fiber_spread(bingham, fd)
        t1 = time.perf_counter()
        logging.info('FS computed in (s): {0}'.format(t1 - t0))
        nib.save(nib.Nifti1Image(fs, bingham_im.affine), args.out_fs)

    if args.out_ff:
        t0 = time.perf_counter()
        logging.info('Computing fiber fraction.')
        ff = compute_fiber_fraction(fd)
        t1 = time.perf_counter()
        logging.info('FS computed in (s): {0}'.format(t1 - t0))
        nib.save(nib.Nifti1Image(ff, bingham_im.affine), args.out_ff)


if __name__ == '__main__':
    main()
