#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute fODF metrics derived for fitting a Bingham distribution
to each fODF lobe, as described in Riffert et al., 2014. Resulting metrics
are fiber density (FD), fiber spread (FS) and fiber fraction (FF)
(Schreiber et al., 2014).

A lobe's FD is the integral of the bingham function on the sphere. It
represents the density of fibers going through a given voxel for a given
fODF lobe (fixel). A lobe's FS is the ratio of its FD on its maximum AFD. It
is at its minimum for a sharp lobe and at its maximum for a wide lobe. A lobe's
FF is the ratio of its FD on the total FD in the voxel.

The Bingham fit is also saved, where each bingham distribution is described
by 9 coefficients (for example, for a maximum number of lobes of 5, the number
of coefficients is 9 x 5 = 45).

Using 12 threads, the execution takes approximately 30 minutes for Bingham
fitting, then 10 minutes for FD estimation for a brain with 1mm isotropic
resolution. Other metrics take less than a second.
"""

import nibabel as nib
import time
import argparse
import logging

from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.reconst.bingham import (bingham_fit_sh,
                                    compute_fiber_density,
                                    compute_fiber_spread,
                                    compute_fiber_fraction)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh', help='Input SH image.')

    p.add_argument('--out_bingham', default='bingham.nii.gz',
                   help='Output bingham functions image. [%(default)s]')
    p.add_argument('--out_fd', default='fd.nii.gz',
                   help='Path to output fiber density. [%(default)s]')
    p.add_argument('--out_fs', default='fs.nii.gz',
                   help='Path to output fiber spread. [%(default)s]')
    p.add_argument('--out_ff', default='ff.nii.gz',
                   help='Path to fiber fraction file. [%(default)s]')

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
                        ' two peaks. [%(default)s]')
    p.add_argument('--max_fit_angle', type=float, default=15.,
                   help='Maximum distance in degrees around a peak direction'
                        ' for fitting the Bingham function. [%(default)s]')
    p.add_argument('--nbr_integration_steps', type=int, default=50,
                   help='Number of integration steps along the theta axis for'
                        ' fiber density estimation. [%(default)s]')

    add_overwrite_arg(p)
    add_verbose_arg(p)
    add_processes_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    outputs = [args.out_bingham, args.out_fd, args.out_fs, args.out_ff]
    assert_inputs_exist(parser, args.in_sh)
    assert_outputs_exist(parser, args, outputs)

    sh_im = nib.load(args.in_sh)
    data = sh_im.get_fdata()

    t0 = time.perf_counter()
    logging.info('Fitting Bingham functions.')
    bingham = bingham_fit_sh(data, args.max_lobes,
                             abs_th=args.at, rel_th=args.rt,
                             min_sep_angle=args.min_sep_angle,
                             max_fit_angle=args.max_fit_angle,
                             nbr_processes=args.nbr_processes)
    t1 = time.perf_counter()
    logging.info('Fitting done in (s): {0}'.format(t1 - t0))
    nib.save(nib.Nifti1Image(bingham, sh_im.affine), args.out_bingham)

    t0 = time.perf_counter()
    logging.info('Computing fiber density.')
    fd = compute_fiber_density(bingham, m=args.nbr_integration_steps,
                               nbr_processes=args.nbr_processes)
    t1 = time.perf_counter()
    logging.info('FD computed in (s): {0}'.format(t1 - t0))
    nib.save(nib.Nifti1Image(fd, sh_im.affine), args.out_fd)

    t0 = time.perf_counter()
    logging.info('Computing fiber spread.')
    fs = compute_fiber_spread(bingham, fd)
    t1 = time.perf_counter()
    logging.info('FS computed in (s): {0}'.format(t1 - t0))
    nib.save(nib.Nifti1Image(fs, sh_im.affine), args.out_fs)

    t0 = time.perf_counter()
    logging.info('Computing fiber fraction.')
    ff = compute_fiber_fraction(fd)
    t1 = time.perf_counter()
    logging.info('FS computed in (s): {0}'.format(t1 - t0))
    nib.save(nib.Nifti1Image(ff, sh_im.affine), args.out_ff)


if __name__ == '__main__':
    main()
