#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import argparse
import logging
import numpy as np

from dipy.core.gradients import (unique_bvals_tolerance, get_bval_indices)
from dipy.io.gradients import read_bvals_bvecs

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             add_b0_thresh_arg, add_skip_b0_check_arg,
                             add_tolerance_arg)
from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              is_normalized_bvecs,
                                              normalize_bvecs)
from scilpy.gradients.gen_gradient_sampling import (generate_gradient_sampling,
                                                    compute_electrostatic_repulsion_energy)
from scilpy.viz.gradients import plot_proj_shell


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_bvals',
                   help='')
    p.add_argument('in_bvecs',
                   help='')

    add_b0_thresh_arg(p)
    add_overwrite_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=True)
    add_tolerance_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Load bvals and bvecs, remove b0s
    bvals, bvecs = read_bvals_bvecs(args.in_bvals, args.in_bvecs)
    args.b0_threshold = check_b0_threshold(bvals.min(),
                                           b0_thr=args.b0_threshold,
                                           skip_b0_check=args.skip_b0_check)
    bvecs = bvecs[bvals > args.b0_threshold]
    bvals = bvals[bvals > args.b0_threshold]

    # Checking bvecs are normalized
    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    ubvals = unique_bvals_tolerance(bvals, tol=args.tolerance)
    list_indices = [get_bval_indices(bvals, shell, tol=args.tolerance)
                    for shell in ubvals]
    nb_shells = len(ubvals)
    nb_dir_per_shell = []
    ubvecs = []
    for indices in list_indices:
        shell_ubvecs = np.unique(bvecs[indices], axis=0)
        if len(indices) != len(shell_ubvecs):
            logging.warning('Some b-vectors have the same direction, which is '
                            'suboptimal. There is most likely a problem with '
                            'the gradient table. Proceeding to validation '
                            'while keeping only unique b-vectors.')
        ubvecs.extend(shell_ubvecs)
        nb_dir_per_shell.append(len(shell_ubvecs))
    ubvecs = np.array(ubvecs)

    scipy_verbose = int(3 - logging.getLogger().getEffectiveLevel()//10)
    opt_bvecs, _ = generate_gradient_sampling(nb_dir_per_shell,
                                              verbose=scipy_verbose)

    # ADD OPTION FOR VISU!!!!!!!
    # plot_proj_shell([ubvecs], use_sym=True)
    # plot_proj_shell([opt_bvecs], use_sym=True)

    energy = compute_electrostatic_repulsion_energy(ubvecs,
                                                    nb_shells=nb_shells,
                                                    nb_points_per_shell=nb_dir_per_shell)
    opt_energy = compute_electrostatic_repulsion_energy(opt_bvecs,
                                                        nb_shells=nb_shells,
                                                        nb_points_per_shell=nb_dir_per_shell)

    print("Input bvecs energy: ", np.round(energy, decimals=2))
    print("Optimal bvecs energy: ", np.round(opt_energy, decimals=2))


if __name__ == "__main__":
    main()
