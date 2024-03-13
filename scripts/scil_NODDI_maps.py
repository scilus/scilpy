#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute NODDI [1] maps using AMICO.
Multi-shell DWI necessary.

Formerly: scil_compute_NODDI.py
"""

import argparse
from contextlib import redirect_stdout
import io
import logging
import os
import sys
import tempfile

import amico
from dipy.io.gradients import read_bvals_bvecs
import numpy as np

from scilpy.io.gradients import fsl2mrtrix
from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             redirect_stdout_c, add_tolerance_arg,
                             add_skip_b0_check_arg)
from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              identify_shells)


EPILOG = """
Reference:
    [1] Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC.
        NODDI: practical in vivo neurite orientation dispersion
        and density imaging of the human brain.
        NeuroImage. 2012 Jul 16;61:1000-16.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, epilog=EPILOG,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dwi',
                   help='DWI file acquired with a NODDI compatible protocol '
                        '(single-shell data not suited).')
    p.add_argument('in_bval',
                   help='b-values filename, in FSL format (.bval).')
    p.add_argument('in_bvec',
                   help='b-vectors filename, in FSL format (.bvec).')

    p.add_argument('--mask',
                   help='Brain mask filename.')
    p.add_argument('--out_dir', default="results",
                   help='Output directory for the NODDI results. '
                        '[%(default)s]')
    add_tolerance_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=False,
                          b0_tol_name='--tolerance')

    g1 = p.add_argument_group(title='Model options')
    g1.add_argument('--para_diff', type=float, default=1.7e-3,
                    help='Axial diffusivity (AD) in the CC. [%(default)s]')
    g1.add_argument('--iso_diff', type=float, default=3e-3,
                    help='Mean diffusivity (MD) in ventricles. [%(default)s]')
    g1.add_argument('--lambda1', type=float, default=5e-1,
                    help='First regularization parameter. [%(default)s]')
    g1.add_argument('--lambda2', type=float, default=1e-3,
                    help='Second regularization parameter. [%(default)s]')

    g2 = p.add_argument_group(title='Kernels options')
    kern = g2.add_mutually_exclusive_group()
    kern.add_argument('--save_kernels', metavar='DIRECTORY',
                      help='Output directory for the COMMIT kernels.')
    kern.add_argument('--load_kernels', metavar='DIRECTORY',
                      help='Input directory where the COMMIT kernels are '
                           'located.')
    g2.add_argument('--compute_only', action='store_true',
                    help='Compute kernels only, --save_kernels must be used.')

    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    # COMMIT has some c-level stdout and non-logging print that cannot
    # be easily stopped. Manual redirection of all printed output
    if args.verbose == "WARNING":
        f = io.StringIO()
        redirected_stdout = redirect_stdout(f)
        redirect_stdout_c() 
    else:
        logging.getLogger().setLevel(logging.getLevelName(args.verbose))
        redirected_stdout = redirect_stdout(sys.stdout)

    # Verifications
    if args.compute_only and not args.save_kernels:
        parser.error('--compute_only must be used with --save_kernels.')

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec],
                        args.mask)

    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       optional=args.save_kernels)

    # Generate a scheme file from the bvals and bvecs files
    bvals, _ = read_bvals_bvecs(args.in_bval, args.in_bvec)
    _ = check_b0_threshold(bvals.min(), b0_thr=args.tolerance,
                           skip_b0_check=args.skip_b0_check)
    shells_centroids, indices_shells = identify_shells(bvals, args.tolerance,
                                                       round_centroids=True)

    nb_shells = len(shells_centroids)
    if nb_shells <= 1:
        raise ValueError("Amico's NODDI works with data with more than one "
                         "shell, but you seem to have single-shell data (we "
                         "found shells {}). Change tolerance if necessary."
                         .format(np.sort(shells_centroids)))

    logging.info('Will compute NODDI with AMICO on {} shells at found at {}.'
                 .format(len(shells_centroids), np.sort(shells_centroids)))

    # Save the resulting bvals to a temporary file
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_scheme_filename = os.path.join(tmp_dir.name, 'gradients.b')
    tmp_bval_filename = os.path.join(tmp_dir.name, 'bval')
    np.savetxt(tmp_bval_filename, shells_centroids[indices_shells],
               newline=' ', fmt='%i')
    fsl2mrtrix(tmp_bval_filename, args.in_bvec, tmp_scheme_filename)

    with redirected_stdout:
        # Load the data
        amico.core.setup()
        ae = amico.Evaluation('.', '.')
        ae.load_data(args.in_dwi, tmp_scheme_filename, mask_filename=args.mask)

        # Compute the response functions
        ae.set_model("NODDI")

        intra_vol_frac = np.linspace(0.1, 0.99, 12)
        intra_orient_distr = np.hstack((np.array([0.03, 0.06]),
                                        np.linspace(0.09, 0.99, 10)))

        ae.model.set(dPar=args.para_diff, dIso=args.iso_diff,
                     IC_VFs=intra_vol_frac, IC_ODs=intra_orient_distr,
                     isExvivo=False)
        ae.set_solver(lambda1=args.lambda1, lambda2=args.lambda2)

        # The kernels are, by default, set to be in the current directory
        # Depending on the choice, manually change the saving location
        if args.save_kernels:
            kernels_dir = args.save_kernels
            regenerate_kernels = True
        elif args.load_kernels:
            kernels_dir = args.load_kernels
            regenerate_kernels = False
        else:
            kernels_dir = os.path.join(tmp_dir.name, 'kernels', ae.model.id)
            regenerate_kernels = True

        ae.set_config('ATOMS_path', kernels_dir)
        ae.set_config('OUTPUT_path', args.out_dir)
        ae.generate_kernels(regenerate=regenerate_kernels)
        if args.compute_only:
            return

        ae.load_kernels()

        # Model fit
        ae.fit()

        # Save the results
        ae.save_results()

    tmp_dir.cleanup()


if __name__ == "__main__":
    main()
