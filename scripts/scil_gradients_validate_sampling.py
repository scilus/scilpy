#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import argparse
import logging
import numpy as np

from dipy.io.gradients import read_bvals_bvecs

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             add_b0_thresh_arg)
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
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    bvals, bvecs = read_bvals_bvecs(args.in_bvals, args.in_bvecs)

    # ADD B0 CHECK

    no_b0_mask = bvals > args.b0_threshold
    bvals = bvals[no_b0_mask]
    bvecs = bvecs[no_b0_mask]
    nb_dir = len(bvals)

    # Add warning when deplicates bvecs.!!!!!!!!!!
    ubvecs, indices = np.unique(bvecs, return_index=True, axis=0) # This sorts the array!
    # ubvals = bvals[indices]

    scipy_verbose = int(3 - logging.getLogger().getEffectiveLevel()//10)
    opt_bvecs, _ = generate_gradient_sampling([nb_dir], verbose=scipy_verbose)

    # ADD OPTION FOR VISU!!!!!!!
    # plot_proj_shell([ubvecs], use_sym=True)
    # plot_proj_shell([opt_bvecs], use_sym=True)

    energy = compute_electrostatic_repulsion_energy(ubvecs)
    opt_energy = compute_electrostatic_repulsion_energy(opt_bvecs)

    print("Input bvecs energy: ", np.round(energy, decimals=2))
    print("Optimal bvecs energy: ", np.round(opt_energy, decimals=2))


if __name__ == "__main__":
    main()
