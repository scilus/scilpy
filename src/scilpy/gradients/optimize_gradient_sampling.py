# -*- coding: utf-8 -*-

import logging

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


def swap_sampling_eddy(bvecs, shell_idx):
    """
    Optimize the bvecs of fixed multi-shell gradient sampling for eddy currents
    correction (fsl EDDY).

    Bruteforce approach to maximally spread the bvec, shell per shell.

    For each shell:
        For each vector:
            1) find the closest neighbor,
            2) flips it,
            3) if global system energy is better, keep it flipped

        repeat until convergence.

    Parameters
    ----------
    bvecs: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs.

    Returns
    -------
    new_bvecs: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs.
    """

    new_bvecs = bvecs.copy()
    nb_points_per_shell = _compute_nb_points_per_shell_from_idx(shell_idx)
    max_nb_iter = 100

    logging.debug("Verifying shells for eddy current optimization")
    for shell in range(len(nb_points_per_shell)):
        # Extract points from shell
        this_shell_idx = shell_idx == shell
        shell_pts = bvecs[this_shell_idx].copy()

        logging.debug('Shell = {}'.format(shell))

        # System energy matrix
        # TODO: test other energy functions such as electron repulsion
        dist = squareform(pdist(shell_pts, 'Euclidean')) \
            + 2 * np.eye(shell_pts.shape[0])

        it = 0
        converged = False
        while (it < max_nb_iter) and not converged:
            converged = True
            # For each bvec on the shell
            for pts_idx in range(len(shell_pts)):
                # Find closest neighbor w.r.t. metric of dist
                to_move = np.argmin(dist[pts_idx])

                # Compute new column of system matrix with flipped to_move
                # point
                new_col = cdist(shell_pts, -shell_pts[None, to_move]).squeeze()

                old_pts_ener = dist[to_move].sum()
                new_pts_ener = new_col.sum()
                if new_pts_ener > old_pts_ener:
                    # Swap sign of point to_move
                    shell_pts[to_move] *= -1
                    dist[:, to_move] = new_col
                    dist[to_move, :] = new_col

                    converged = False

                    logging.debug('Swapped {} ({:.2f} --> {:.2f})'
                                  .format(to_move, old_pts_ener, new_pts_ener))

            it += 1

        new_bvecs[this_shell_idx] = shell_pts

    return new_bvecs, shell_idx


def _compute_nb_points_per_shell_from_idx(shell_idx):
    """
    Recover number of points per shell from point-wise shell index.

    Parameters
    ----------
    shell_idx: numpy.array
        Shell index of gradient sampling.

    Return
    ------
    Ks: list
        number of samples for each shell, starting from lowest.
    """
    nb_shells = len(set(shell_idx))

    nb_points_per_shell = []
    for idx in range(nb_shells):
        nb_points_per_shell.append(np.sum(shell_idx == idx))

    return nb_points_per_shell


def add_b0s_to_bvecs(bvecs, shell_idx, start_b0=True, b0_every=None,
                     finish_b0=False):
    """
    Add interleaved b0s to gradient sampling.

    Parameters
    ----------
    bvecs: numpy.array,
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs.
    start_b0: bool
        Option to add a b0 at the beginning.
    b0_every: integer or None
        Final gradient sampling will have a b0 every b0_every samples.
        (start_b0 must be true)
    finish_b0: bool
        Option to add a b0 as last sample.

    Return
    ------
    new_bvecs: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs. Vectors with shells of value -1 are b0 vectors.
    nb_new_b0s: int
        The number of b0s interleaved.
    """
    new_bvecs = []
    new_shell_idx = []

    # Only a b0 at the beginning.
    # Same as a b0 every n with n > nb_samples
    # By default, we do add a b0
    nb_points_total = len(shell_idx)

    # Prepare b0_every
    if b0_every is not None:
        if not start_b0:
            raise ValueError("Can't add a b0 every {} values with option "
                             "start_b0 at False.".format(b0_every))
    elif start_b0:
        # Setting b0_every to one more than the total number creates the right
        # result.
        b0_every = nb_points_total + 1

    if b0_every is not None:
        for idx in range(nb_points_total):
            if not idx % (b0_every - 1):
                # insert b0
                new_bvecs.append(np.array([0.0, 0.0, 0.0]))
                new_shell_idx.append(-1)  # Shell -1 ==> means b0.

            # Add pre-defined points.
            new_bvecs.append(bvecs[idx])
            new_shell_idx.append(shell_idx[idx])

    if finish_b0 and (new_shell_idx[-1] != -1):
        # insert b0
        new_bvecs.append(np.array([0.0, 0.0, 0.0]))
        new_shell_idx.append(-1)

    nb_new_b0s = len(new_shell_idx) - shell_idx.shape[0]

    return np.asarray(new_bvecs), np.asarray(new_shell_idx), nb_new_b0s


def correct_b0s_philips(bvecs, shell_idx):
    """
    Replace the [0.0, 0.0, 0.0] value of b0s bvecs by existing bvecs in the
    gradient sampling, except possibly the first one.

    This is useful because Recon 1.0 of Philips allocates memory proportional
    to (total nb. of diff. bvals) x (total nb. diff. bvecs) and we can't leave
    multiple b0s with b-vector [0.0, 0.0, 0.0] and b-value 0 because
    (b-vector, b-value) pairs have to be unique.

    Parameters
    ----------
    bvecs: numpy.array
        bvecs normalized to 1
    shell_idx: numpy.array
        Shell index for bvecs. Vectors with shells of value -1 are b0 vectors.

    Return
    ------
    new_bvecs: numpy.array
        bvecs normalized to 1. b0 vectors are now impossible to know as they
        are replaced by random values from another vector.
    shell_idx: numpy.array
        Shell index for bvecs. b0 vectors still have shells of value -1.
    """

    new_bvecs = bvecs.copy()

    # We could replace by a random value, but (we think that... to verify?)
    # the machine is more efficient if we copy the previous gradient; the
    # machine then does have to change its parameters between images.

    # 1. By default, other shells should already be ok (never twice the same
    # gradients per shell.)
    # 2. Assume non-collinearity of non-b0s bvecs (i.e. Caruyer sampler type)
    # between shells. Could be verified?
    # 3. Assume that we never have two b0s one after the other. This is how we
    # build them in our scripts.

    new_bvecs[np.where(shell_idx == -1)[0][1:]] \
        = new_bvecs[np.where(shell_idx == -1)[0][1:] - 1]

    logging.info('Done adapting b0s for Philips scanner.')

    return new_bvecs, shell_idx


def compute_min_duty_cycle_bruteforce(bvecs, shell_idx, bvals, ker_size=10,
                                      nb_iter=100000, rand_seed=0):
    """
    Optimize the ordering of non-b0 samples to optimize gradient duty-cycle.

    Philips scanner (and other) will find the peak power requirements with its
    duty cycle model (this is an approximation) and increase the TR accordingly
    to the hardware needs. This minimizes this effect by:

    1) Randomly permuting the non-b0s samples
    2) Finding the peak X, Y, and Z amplitude with a sliding-window
    3) Computing the peak power needed as max(peak_x, peak_y, peak_z)
    4) Keeping the permutation yielding the lowest peak power

    Parameters
    ----------
    bvecs: numpy.array
        bvecs normalized to 1
    shell_idx: numpy.array
        Shell index for bvecs.
    bvals: list
        increasing bvals, b0 last.
    ker_size: int
        kernel size for the sliding window.
    nb_iter: int
        number of bruteforce iterations.
    rand_seed: int
        seed for the random permutations.

    Return
    ------
    new_bvecs: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs.
    """

    logging.debug('Shuffling Data (N_iter = {}, \
                                   ker_size = {})'.format(nb_iter, ker_size))

    non_b0s_mask = shell_idx != -1
    N_dir = non_b0s_mask.sum()

    sqrt_val = np.sqrt(np.array([bvals[idx] for idx in shell_idx]))
    q_scheme = np.abs(bvecs * sqrt_val[:, None])

    q_scheme_current = q_scheme.copy()

    ordering_best = np.arange(N_dir)
    power_best = compute_peak_power(q_scheme_current, ker_size=ker_size)
    logging.info("Duty cycle: initial peak power = {}".format(power_best))

    np.random.seed(rand_seed)

    for it in range(nb_iter):
        if not it % np.ceil(nb_iter / 10.):
            logging.debug('Iter {} / {}  : {}'.format(it, nb_iter, power_best))

        ordering_current = np.random.permutation(N_dir)
        q_scheme_current[non_b0s_mask] \
            = q_scheme[non_b0s_mask][ordering_current]

        power_current = compute_peak_power(q_scheme_current, ker_size=ker_size)

        if power_current < power_best:
            ordering_best = ordering_current.copy()
            power_best = power_current

    logging.info('Duty cycle optimization finished ({} iterations). '
                 'Final peak power: {}'.format(nb_iter, power_best))

    new_bvecs = bvecs.copy()
    new_bvecs[non_b0s_mask] = bvecs[non_b0s_mask][ordering_best]

    new_shell_idx = shell_idx.copy()
    new_shell_idx[non_b0s_mask] = shell_idx[non_b0s_mask][ordering_best]

    return new_bvecs, new_shell_idx


def compute_peak_power(q_scheme, ker_size=10):
    """
    Function suggested by Guillaume Gilbert.

    Optimize the diffusion gradient table by minimizing the maximum gradient
    load on any of the 3 axes over a preset temporal window (i.e. successive
    b-vectors).

    In short, we want to avoid using the same gradient axis (x, y, or z)
    intensely for many successive b-vectors.

    Parameters
    ------
    q_scheme: nd.array
        Scheme of acquisition.
    ker_size: int
        Kernel size (default=10).

    Return
    ------
        Max peak power from q_scheme.
    """

    # Using a filter of ones = moving average.
    ker = np.ones(ker_size)

    # Note: np.convolve inverses the filter
    pow_x = np.convolve(q_scheme[:, 0], ker, 'full')[:-(ker_size-1)]
    pow_y = np.convolve(q_scheme[:, 1], ker, 'full')[:-(ker_size-1)]
    pow_z = np.convolve(q_scheme[:, 2], ker, 'full')[:-(ker_size-1)]

    max_pow_x = np.max(pow_x)
    max_pow_y = np.max(pow_y)
    max_pow_z = np.max(pow_z)

    return np.max([max_pow_x, max_pow_y, max_pow_z])


def compute_bvalue_lin_q(bmin=0.0, bmax=3000.0, nb_of_b_inside=2,
                         exclude_bmin=True):
    """
    Compute bvals linearly distributed in q-value in the interval [bmin, bmax].
    This leads to sqrt(b_values) linearly distributed.

    Parameters
    ----------
    bmin: float
        Minimum b-value, lower b-value bounds.
    bmax: float
        Maximum b-value, upper b-value bounds.
    nb_of_b_inside: int
        number of b-value excluding bmin and bmax.
    exclude_bmin: bool
        exclude bmin from the interval, useful if bmin = 0.0.

    Return
    ------
    bvals: list
        increasing bvals.
    """

    bvals = list(np.linspace(np.sqrt(bmin),
                             np.sqrt(bmax),
                             nb_of_b_inside + 2)**2)
    if exclude_bmin:
        bvals = bvals[1:]

    logging.info('bvals linear in q: {}'.format(bvals))

    return bvals


def compute_bvalue_lin_b(bmin=0.0, bmax=3000.0, nb_of_b_inside=2,
                         exclude_bmin=True):
    """
    Compute bvals linearly distributed in b-value in the interval [bmin, bmax].

    Parameters
    ----------
    bmin: float
        Minimum b-value, lower b-value bounds.
    bmax: float
        Maximum b-value, upper b-value bounds.
    nb_of_b_inside: int
        number of b-value excluding bmin and bmax.
    exclude_bmin: boolean
        exclude bmin from the interval, useful if bmin = 0.0.

    Return
    ------
    bvals: list
        increasing bvals.
    """

    bvals = list(np.linspace(bmin, bmax, nb_of_b_inside + 2))
    if exclude_bmin:
        bvals = bvals[1:]

    logging.info('bvals linear in b: {}'.format(bvals))

    return bvals
