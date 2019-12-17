#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np

from scilpy.io.utils import (assert_outputs_exist,
                             add_overwrite_arg, add_verbose_arg)
from scilpy.samplingscheme.gen_scheme import gen_scheme
from scilpy.samplingscheme.optimize_scheme import (add_b0s,
                                                   add_bvalue_b0,
                                                   compute_bvalue_lin_b,
                                                   compute_bvalue_lin_q,
                                                   correct_b0s_philips,
                                                   compute_min_duty_cycle_bruteforce,
                                                   swap_sampling_eddy)
from scilpy.samplingscheme.save_scheme import (save_scheme_bvecs_bvals,
                                               save_scheme_caru,
                                               save_scheme_philips,
                                               save_scheme_siemens,
                                               save_scheme_mrtrix)


DESCRIPTION = """
Generate multi-shell sampling schemes with various processing to accelerate
acquisition and help artefact correction.

Multi-shell schemes are generated as in [1], the bvecs are then flipped
to maximize spread for eddy current correction, b0s are interleaved
at equal spacing and the non-b0 samples are finally shuffled
to minimize the total diffusion gradient amplitude over a few TR.
    """

EPILOG = """
References: [1] Emmanuel Caruyer, Christophe Lenglet, Guillermo Sapiro,
Rachid Deriche. Design of multishell sampling schemes with uniform coverage
in diffusion MRI. Magnetic Resonance in Medicine, Wiley, 2013, 69 (6),
pp. 1534-1540. <http://dx.doi.org/10.1002/mrm.24736>
    """


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=DESCRIPTION,
        epilog=EPILOG)
    p._optionals.title = "Options and Parameters"

    p.add_argument('nb_samples',
                   type=int, nargs='+',
                   help='Number of samples on each shells. If multishell, '
                        'provide a number per shell.')
    p.add_argument('outfile',
                   help='Sampling scheme output filename (don\'t '
                        'include extension).')

    p.add_argument('--eddy',
                   action='store_true',
                   help='Apply eddy optimization.\nB-vectors are flipped '
                        'to be well spread without symmetry. [%(default)s]')
    p.add_argument('--duty',
                   action='store_true',
                   help='Apply duty cycle optimization. '
                        '\nB-vectors are shuffled to reduce consecutive '
                        'colinearity in the samples. [%(default)s]')
    p.add_argument('--b0inter',
                   dest='b0_every', type=int, default=-1,
                   help='Interleave a b0 every b0_every. \nNo b0 if 0. '
                        '\nOnly 1 b0 at beginning if > number of samples '
                        'or negative. [%(default)s]')
    p.add_argument('--b0end',
                   action='store_true', dest='b0_end',
                   help='Add a b0 as last sample. [%(default)s]')
    p.add_argument('--b0value',
                   dest='b0_value', type=float, default=0.0,
                   help='b-value of the b0s. [%(default)s]')

    bvals_group = p.add_mutually_exclusive_group(required=True)
    bvals_group.add_argument('--bvals',
                             type=float, nargs='+', metavar='bvals',
                             help='bval of each non-b0 shell.')
    bvals_group.add_argument('--blinmax',
                             dest='b_lin_max', type=float,
                             help='b-max for linear bval distribution '
                                  'in *b*. [replaces -bvals]')
    bvals_group.add_argument('--qlinmax',
                             dest='q_lin_max', type=float,
                             help='b-max for linear bval distribution '
                                  'in *q*. [replaces -bvals]')

    g1 = p.add_argument_group(title='Save as')
    g1.add_argument('--caru',
                    action='store_true',
                    help='Save in caruyer format (.caru). [%(default)s]')
    g1.add_argument('--phil',
                    action='store_true',
                    help='Save in Philips format (.txt). [%(default)s]')
    g1.add_argument('--fsl',
                    action='store_true',
                    help='Save in FSL format (.bvecs/.bvals). [%(default)s]')
    g1.add_argument('--siemens',
                    action='store_true',
                    help='Save in Siemens format (.dvs). [%(default)s]')
    g1.add_argument('--mrtrix',
                    action='store_true',
                    help='Save in MRtrix format (.b). [%(default)s]')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():

    parser = _build_args_parser()
    args = parser.parse_args()

    caru = args.caru
    phil = args.phil
    fsl = args.fsl
    siemens = args.siemens
    mrtrix = args.mrtrix

    if not (caru or phil or fsl or siemens or mrtrix):
        parser.error('Select at least one save format.')
        return

    logging.basicConfig(level=logging.INFO)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    Ks = args.nb_samples
    eddy = args.eddy
    duty = args.duty

    # Total number of samples
    K = np.sum(Ks)
    # Number of non-b0 shells
    S = len(Ks)

    b0_every = args.b0_every
    b0_end = args.b0_end
    b0_value = args.b0_value

    # Only a b0 at the beginning
    if (b0_every > K) or (b0_every < 0):
        b0_every = K + 1

    # Compute bval list
    if args.bvals is not None:
        bvals = args.bvals
    elif args.b_lin_max is not None:
        bvals = compute_bvalue_lin_b(bmin=0.0, bmax=args.b_lin_max,
                                     nb_of_b_inside=S - 1, exclude_bmin=True)
    elif args.q_lin_max is not None:
        bvals = compute_bvalue_lin_q(bmin=0.0, bmax=args.q_lin_max,
                                     nb_of_b_inside=S - 1, exclude_bmin=True)
    # Add b0 b-value
    if b0_every != 0:
        bvals = add_bvalue_b0(bvals, b0_value=b0_value)

    outfile = args.outfile

    # Scheme generation
    points, shell_idx = gen_scheme(Ks, verbose=int(
        3 - logging.getLogger().getEffectiveLevel()//10))

    # eddy current optimization
    if eddy:
        points, shell_idx = swap_sampling_eddy(points, shell_idx)

    # Adding interleaved b0s
    if b0_every != 0:
        points, shell_idx = add_b0s(
            points, shell_idx, b0_every=b0_every, finish_b0=b0_end)

    # duty cycle optimization
    if duty:
        points, shell_idx = compute_min_duty_cycle_bruteforce(
            points, shell_idx, bvals)

    # Save the sampling scheme
    if caru:
        assert_outputs_exist(parser, args, outfile + '.caru')
        save_scheme_caru(points, shell_idx, filename=outfile)

    if fsl:
        assert_outputs_exist(parser, args, [outfile + '.bvecs',
                                            outfile + '.bvals'])
        save_scheme_bvecs_bvals(points, shell_idx, bvals, filename=outfile)

    if siemens:
        assert_outputs_exist(parser, args, outfile + '.dvs')
        save_scheme_siemens(points, shell_idx, bvals, filename=outfile)

    if mrtrix:
        assert_outputs_exist(parser, args, outfile + '.b')
        save_scheme_mrtrix(points, shell_idx, bvals, filename=outfile)

    if phil:
        # Correcting bvecs for b0s
        if b0_every != 0:
            points, shell_idx = correct_b0s_philips(points, shell_idx)

        assert_outputs_exist(parser, args, outfile + '.txt')
        save_scheme_philips(points, shell_idx, bvals, filename=outfile)


if __name__ == "__main__":
    main()
