# -*- coding: utf-8 -*-

import numpy as np

from scilpy.samplingscheme.multiple_shell_energy import (compute_weights,
                                                         multiple_shell)


def generate_scheme(nb_samples, verbose=1):
    """
    Wrapper code to generate sampling scheme from Caruyer's
    multiple_shell_energy.py

    Generate the bvecs of a multiple shell sampling scheme using generalized
    Jones electrostatic repulsion.

    Parameters
    ----------
    nb_samples: list
        number of samples for each shell, starting from lowest.
    verbose: 0 = silent, 1 = summary upon completion, 2 = print iterations

    Return
    ------
    points: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs in points.
    """

    S = len(nb_samples)

    # Groups of shells and relative coupling weights
    shell_groups = ()
    for i in range(S):
        shell_groups += ([i],)

    shell_groups += (list(range(S)),)
    alphas = len(shell_groups) * (1.0,)
    weights = compute_weights(S, nb_samples, shell_groups, alphas)

    # Where the optimized sampling scheme is computed
    # max_iter hardcoded to fit default Caruyer's value
    points = multiple_shell(S, nb_samples, weights,
                            max_iter=100, verbose=verbose)

    shell_idx = np.repeat(range(S), nb_samples)

    return points, shell_idx
