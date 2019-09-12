from __future__ import division

import numpy as np

from scilpy.samplingscheme.multiple_shell_energy import compute_weights, multiple_shell

def gen_scheme(Ks, verbose = 1):
    """
    Wrapper code to generate sampling scheme from Caruyer's multiple_shell_energy.py

    Generate the bvecs of a multiple shell sampling scheme using generalized Jones 
    electrostatic repulsion.

    Parameters
    ----------
    Ks: list, number of samples for each shell, starting from lowest.
    verbose: 0 = silent, 1 = summary upon completion, 2 = print iterations

    Return
    ------
    points: numpy.array, b-vectors normalized to 1.
    shell_idx: numpy.array, Shell index for bvecs in points.

    """

    S = len(Ks)

    # Groups of shells and relative coupling weights
    shell_groups = ()
    for i in range(S):
        shell_groups+=([i],)

    shell_groups += (range(S),)
    alphas = len(shell_groups) * (1.0,)
    weights = compute_weights(S, Ks, shell_groups, alphas)

    # Where the optimized sampling scheme is computed
    # max_iter hardcoded to fit default Caruyer's value
    points = multiple_shell(S, Ks, weights, max_iter=100, verbose = verbose)

    shell_idx = []
    for idx in range(S):
        for nb_pts in range(Ks[idx]):
            shell_idx.append(idx)
    shell_idx = np.array(shell_idx)

    return points, shell_idx
