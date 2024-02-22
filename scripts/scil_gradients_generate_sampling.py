#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate multi-shell gradient sampling with various processing options. Helps
accelerate gradients, optimize duty cycle and avoid artefacts.

Multi-shell gradient sampling is generated as in [1]. The bvecs are then
flipped to maximize spread for eddy current correction, b0s are interleaved at
equal spacing and the non-b0 samples are finally shuffled to minimize the total
diffusion gradient amplitude over a few TR.

Formerly: scil_generate_gradient_sampling.py
"""

import argparse
import logging
import numpy as np
import os

from scilpy.io.utils import (
    add_overwrite_arg, add_verbose_arg, assert_outputs_exist)
from scilpy.gradients.gen_gradient_sampling import (
    generate_gradient_sampling)
from scilpy.gradients.optimize_gradient_sampling import (
    add_b0s_to_bvecs, compute_bvalue_lin_b, compute_bvalue_lin_q,
    compute_min_duty_cycle_bruteforce, correct_b0s_philips, swap_sampling_eddy)
from scilpy.io.gradients import (
    save_gradient_sampling_fsl, save_gradient_sampling_mrtrix)


EPILOG = """
References: [1] Emmanuel Caruyer, Christophe Lenglet, Guillermo Sapiro,
Rachid Deriche. Design of multishell gradient sampling with uniform coverage
in diffusion MRI. Magnetic Resonance in Medicine, Wiley, 2013, 69 (6),
pp. 1534-1540. <http://dx.doi.org/10.1002/mrm.24736>
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
        epilog=EPILOG)
    p.add_argument('nb_samples_per_shell', type=int, nargs='+',
                   help='Number of samples on each non b0 shell. \n'
                        'If multishell, provide a number per shell.')
    p.add_argument('out_basename',
                   help='Gradient sampling output basename (don\'t '
                        'include extension).\n'
                        'Please add options --fsl and/or --mrtrix below.')

    p.add_argument('--eddy', action='store_true',
                   help='If set, we apply eddy optimization.\nB-vectors are '
                        'flipped to be well spread without symmetry.')
    p.add_argument('--duty', action='store_true',
                   help='If set, we apply duty cycle optimization. '
                        '\nB-vectors are shuffled to reduce consecutive '
                        'colinearity in the samples. [%(default)s]')

    g = p.add_argument_group("b0 acquisitions",
                             "Default if you add no option is to have a b0 "
                             "at the start.")
    gg = g.add_mutually_exclusive_group()
    gg.add_argument('--no_b0_start',
                    help="If set, do not add a b0 at the beginning. ")
    gg.add_argument('--b0_every', type=int,
                    help='Interleave a b0 every n=b0_every values. Starts '
                         'after the first b0 \n(cannot be used with '
                         '--no_b0_start). Must be an integer >= 1.')
    g.add_argument('--b0_end', action='store_true',
                   help='If set, adds a b0 as last sample.')
    g.add_argument('--b0_value', type=float, default=0.0,
                   help='b-value of the b0s. [%(default)s]')
    g.add_argument('--b0_philips', action='store_true',
                   help='If set, replaces values of b0s bvecs by existing '
                        'bvecs for Philips handling.')

    g = p.add_argument_group("Non-b0 acquisitions")
    bvals_group = g.add_mutually_exclusive_group(required=True)
    bvals_group.add_argument('--bvals', type=float, nargs='+', metavar='bvals',
                             help='bval of each non-b0 shell.')
    bvals_group.add_argument('--b_lin_max', type=float,
                             help='b-max for linear bval distribution '
                                  'in *b*.')
    bvals_group.add_argument('--q_lin_max', type=float,
                             help='b-max for linear bval distribution '
                                  'in *q*; \nthe square root of b-values will '
                                  'be linearly distributed..')

    g1 = p.add_argument_group(title='Save as')
    g1 = g1.add_mutually_exclusive_group(required=True)
    g1.add_argument('--fsl', action='store_true',
                    help='Save in FSL format (.bvec/.bval).')
    g1.add_argument('--mrtrix', action='store_true',
                    help='Save in MRtrix format (.b).')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # ---- Checks
    out_basename, _ = os.path.splitext(args.out_basename)

    if args.fsl:
        out_filename = [out_basename + '.bval', out_basename + '.bvec']
    else:  # mrtrix
        out_filename = out_basename + '.b'

    assert_outputs_exist(parser, args, out_filename)

    nb_shells = len(args.nb_samples_per_shell)
    if args.bvals is not None:
        unique_bvals = np.unique(args.bvals)
        if len(unique_bvals) != nb_shells:
            parser.error('You have provided {} shells '.format(nb_shells) +
                         'but {} unique bvals.'.format(len(unique_bvals)))
    if args.b0_every is not None and args.b0_every <= 0:
        parser.error("--b0_every must be an integer > 0.")

    # ---- b-vectors generation
    # Non-b0 samples: gradient sampling generation
    # Generates the b-vectors only. We will set their b-values after.
    logging.info("Generating b-vectors.")
    scipy_verbose = int(3 - logging.getLogger().getEffectiveLevel()//10)
    bvecs, shell_idx = generate_gradient_sampling(
        args.nb_samples_per_shell, verbose=scipy_verbose)

    # eddy current optimization
    if args.eddy:
        logging.info("Optimizing values for Eddy current.")
        bvecs, shell_idx = swap_sampling_eddy(bvecs, shell_idx)

    # ---- b-values
    logging.info("Preparing b-values.")
    if args.bvals is not None:
        bvals = args.bvals
    elif args.b_lin_max is not None:
        bvals = compute_bvalue_lin_b(
            bmin=0.0, bmax=args.b_lin_max, nb_of_b_inside=nb_shells - 1,
            exclude_bmin=True)
    else:  # args.q_lin_max:
        bvals = compute_bvalue_lin_q(
            bmin=0.0, bmax=args.q_lin_max,  nb_of_b_inside=nb_shells - 1,
            exclude_bmin=True)

    # ---- Adding b0s
    b0_start = not args.no_b0_start
    add_at_least_a_b0 = b0_start or (args.b0_every is not None) or args.b0_end
    if add_at_least_a_b0:
        bvals.append(args.b0_value)
        bvecs, shell_idx, nb_b0s = add_b0s_to_bvecs(bvecs, shell_idx,
                                                    start_b0=b0_start,
                                                    b0_every=args.b0_every,
                                                    finish_b0=args.b0_end)
        logging.info('   Interleaved {} b0s'.format(nb_b0s))
    else:
        logging.info("   Careful! No b0 added!")

    # duty cycle optimization
    if args.duty:
        logging.info("Optimizing the ordering of non-b0s samples to optimize "
                     "gradient duty-cycle.")
        bvecs, shell_idx = compute_min_duty_cycle_bruteforce(
            bvecs, shell_idx, bvals)

    # Correcting b0s bvecs for Philips
    if args.b0_philips and np.sum(shell_idx == -1) > 1:
        logging.info("Correcting b0 vectors for Philips")
        bvecs, shell_idx = correct_b0s_philips(bvecs, shell_idx)

    logging.info("Done. Saving in {}".format(out_filename))

    if args.fsl:
        save_gradient_sampling_fsl(bvecs, shell_idx, bvals,
                                   out_filename[0], out_filename[1])
    else:  # args.mrtrix:
        if not bvecs.shape[0] == 3:
            bvecs = bvecs.transpose()
            save_gradient_sampling_mrtrix(bvecs, shell_idx, bvals,
                                          filename=out_filename)
        else:
            raise ValueError("Expecting bvecs.shape[0] to be different than "
                             "3 but not the case. Error in scilpy's code? "
                             "What is the case here?")


if __name__ == "__main__":
    main()
