from __future__ import division

import logging
import numpy as np

from scipy.spatial.distance import cdist, pdist, squareform


# TODO: make this robust to having b0s
def swap_sampling_eddy(points, shell_idx, verbose = 1):
    """
    Optimize the b-vectors of fixed multi-shell scheme for eddy 
    currents correction (fsl EDDY).

    Bruteforce approach to maximally spread the b-vector, 
    shell per shell.

    For each shell:
        For each vector:
            1) find the closest neighbor,
            2) flips it,
            3) if global system energy is better, keep it flipped
    
        repeat until convergence.


    Parameters
    ----------
    points: numpy.array, b-vectors normalized to 1.
    shell_idx: numpy.array, Shell index for bvecs in points.
    verbose: 0 = silent, 1 = summary upon completion, 2 = print iterations.

    Return
    ------
    points: numpy.array, b-vectors normalized to 1.
    shell_idx: numpy.array, Shell index for bvecs in points.

    """

    new_points = points.copy()

    Ks = compute_ks_from_shell_idx(shell_idx)

    maxIter = 100

    for shell in range(len(Ks)):
        # Extract points from shell
        shell_pts = points[shell_idx == shell].copy()

        logging.debug('Shell = {}'.format(shell))

        # System energy matrix
        # TODO: test other energy functions such as electron repulsion
        dist = squareform(pdist(shell_pts, 'Euclidean')) + 2 * np.eye(shell_pts.shape[0])

        it = 0
        converged = False

        while (it < maxIter) and not converged:
            converged = True
            # For each b-vector on the shell
            for pts_idx in range(len(shell_pts)):
                # Find closest neighbor w.r.t. metric of dist
                toMove = np.argmin(dist[pts_idx])
                # Compute new column of system matrix with flipped toMove point
                new_col = cdist(shell_pts, -shell_pts[None, toMove]).squeeze()

                old_pts_ener = dist[toMove].sum()
                new_pts_ener = new_col.sum()

                if new_pts_ener > old_pts_ener:
                    # Swap sign of point toMove
                    shell_pts[toMove] *= -1
                    dist[:, toMove] = new_col
                    dist[toMove, :] = new_col

                    converged = False

                    logging.debug('Swapped {} ({:.2f} --> {:.2f})'.format(toMove, old_pts_ener, new_pts_ener))

            it += 1

        new_points[shell_idx == shell] = shell_pts

    logging.info('Eddy current swap optimization finished.')

    return new_points, shell_idx


def compute_ks_from_shell_idx(shell_idx):
    """
    Recover number of points per shell from point-wise shell index.


    Parameters
    ----------
    shell_idx: numpy.array, Shell index of sampling scheme.

    Return
    ------
    Ks: list, number of samples for each shell, starting from lowest.

    """
    K = len(set(shell_idx))

    Ks = []
    for idx in range(K):
        Ks.append(np.sum(shell_idx == idx))

    return Ks


def add_b0s(points, shell_idx, b0_every = 10, finish_b0 = False, verbose = 1):
    """
    Add interleaved b0s to sampling scheme.

    Parameters
    ----------
    points: numpy.array, b-vectors normalized to 1.
    shell_idx: numpy.array, Shell index for bvecs in points.
    b0_every: integer, final scheme will have a b0 every b0_every samples
    finish_b0: boolean, Option to add a b0 as last sample.
    verbose: 0 = silent, 1 = summary upon completion, 2 = print iterations.

    Return
    ------
    points: numpy.array, b-vectors normalized to 1.
    shell_idx: numpy.array, Shell index for bvecs in points.

    """

    new_points = []
    new_shell_idx = []

    for idx in range(shell_idx.shape[0]):
        if not idx%(b0_every - 1):
            # insert b0
            new_points.append(np.array([0.0, 0.0, 0.0]))
            new_shell_idx.append(-1)

        new_points.append(points[idx])
        new_shell_idx.append(shell_idx[idx])

    if finish_b0 and (new_shell_idx[-1] != -1):
        # insert b0
        new_points.append(np.array([0.0, 0.0, 0.0]))
        new_shell_idx.append(-1)

    logging.info('Interleaved {} b0s'.format(len(new_shell_idx) - shell_idx.shape[0]))

    return np.array(new_points), np.array(new_shell_idx)


def correct_b0s_philips(points, shell_idx, verbose = 1):
    """
    Replace the [0.0, 0.0, 0.0] value of b0s b-vectors
    by existing b-vectors in the sampling scheme.

    This is useful because Recon 1.0 of Philips allocates memory proportional to
    (total nb. of diff. b-values) x (total nb. diff. b-vectors)
    and we can't leave multiple b0s with b-vector [0.0, 0.0, 0.0] and b-value 0
    because (b-vector, b-value) pairs have to be unique.

    Parameters
    ----------
    points: numpy.array, b-vectors normalized to 1.
    shell_idx: numpy.array, Shell index for bvecs in points.
    verbose: 0 = silent, 1 = summary upon completion, 2 = print iterations.

    Return
    ------
    points: numpy.array, b-vectors normalized to 1.
    shell_idx: numpy.array, Shell index for bvecs in points.

    """

    new_points = points.copy()

    non_b0_pts = points[np.where(shell_idx != -1)]

    it = 0
    # Assume non-collinearity of non-b0s b-vectors (i.e. Caruyer sampler type)
    for idx in np.where(shell_idx == -1)[0]:
        new_points[idx] = non_b0_pts[it]
        it += 1

    logging.info('Done adapting b0s for Philips scanner.')

    return new_points, shell_idx


def min_duty_cycle_bruteforce(points, shell_idx, bvalues, ker_size = 10, Niter = 100000, 
                              verbose = 1, plotting = False, rand_seed = 0):
    """
    Optimize the ordering of non-b0s sample to optimize gradient duty-cycle.

    Philips scanner (and other) will find the peak power requirements with its 
    duty cycle model (this is an approximation) and increase the TR accordingly 
    to the hardware needs. This minimize this effects by:

    1) Randomly permuting the non-b0s samples
    2) Finding the peak X, Y, and Z amplitude with a sliding-window
    3) Compute peak power needed as max(peak_x, peak_y, peak_z)
    4) Keeps the permutation yielding the lowest peak power


    Parameters
    ----------
    points: numpy.array, b-vectors normalized to 1.
    shell_idx: numpy.array, Shell index for bvecs in points.
    b_values: list, increasing b-values, b0 last.
    ker_size: integer, kernel size for the sliding window.
    Niter: integer, number of bruteforce iterations.
    verbose: 0 = silent, 1 = summary upon completion, 2 = print iterations.
    plotting: boolean, plot the energy at each iteration. 
    rand_seed: integer, seed for the random permutations.

    Return
    ------
    points: numpy.array, b-vectors normalized to 1.
    shell_idx: numpy.array, Shell index for bvecs in points.

    """

    logging.debug('Shuffling Data (N_iter = {}, ker_size = {})'.format(Niter, ker_size))

    if plotting:
        store_best_value = []
    

    non_b0s_mask = shell_idx != -1
    N_dir = non_b0s_mask.sum()

    q_scheme = np.abs(points * np.sqrt(np.array([bvalues[idx] for idx in shell_idx]))[:, None])

    q_scheme_current = q_scheme.copy()

    ordering_best = np.arange(N_dir)
    power_best = _peak_power(q_scheme_current, ker_size = ker_size)

    if plotting:
        store_best_value.append((0, power_best))

    np.random.seed(rand_seed)

    for it in range(Niter):
        if not it%np.ceil(Niter/10.):
            logging.debug('Iter {} / {}  : {}'.format(it, Niter, power_best))

        ordering_current = np.random.permutation(N_dir)
        q_scheme_current[non_b0s_mask] = q_scheme[non_b0s_mask][ordering_current]

        power_current = _peak_power(q_scheme_current, ker_size = ker_size)

        if power_current < power_best:
            ordering_best = ordering_current.copy()
            power_best = power_current

            if plotting:
                store_best_value.append((it+1, power_best))

    logging.debug('Iter {} / {}  : {}'.format(Niter, Niter, power_best))

    logging.info('Duty cycle optimization finished.')

    if plotting:
        store_best_value = np.array(store_best_value)
        import pylab as pl
        pl.plot(store_best_value[:,0], store_best_value[:,1], '-o')
        pl.show()


    new_points = points.copy()
    new_points[non_b0s_mask] = points[non_b0s_mask][ordering_best]

    new_shell_idx = shell_idx.copy()
    new_shell_idx[non_b0s_mask] = shell_idx[non_b0s_mask][ordering_best]

    return new_points, new_shell_idx


def _peak_power(q_scheme, ker_size = 10):
    # Note: np.convolve inverses the filter
    ker = np.ones(ker_size)

    pow_x = np.convolve(q_scheme[:, 0], ker, 'full')[:-(ker_size-1)]
    pow_y = np.convolve(q_scheme[:, 1], ker, 'full')[:-(ker_size-1)]
    pow_z = np.convolve(q_scheme[:, 2], ker, 'full')[:-(ker_size-1)]

    max_pow_x = np.max(pow_x)
    max_pow_y = np.max(pow_y)
    max_pow_z = np.max(pow_z)

    return np.max([max_pow_x, max_pow_y, max_pow_z])


def bvalue_lin_q(bmin = 0.0, bmax = 3000.0, nb_of_b_inside = 2, exclude_bmin = True, verbose = 1):
    """
    Compute b-values linearly distributed in q-value in the 
    interval [bmin, bmax].

    Parameters
    ----------
    bmin: float, Minimum b-value, lower b-value bounds.
    bmax: float, Maximum b-value, upper b-value bounds.
    nb_of_b_inside: integer, number of b-value excluding bmin and bmax.
    exclude_bmin: boolean, exclude bmin from the interval, useful if bmin = 0.0.
    verbose: 0 = silent, 1 = summary upon completion, 2 = print iterations.

    Return
    ------
    b_values: list, increasing b-values.

    """

    b_values = list(np.linspace(np.sqrt(bmin), np.sqrt(bmax), nb_of_b_inside + 2)**2)
    if exclude_bmin:
        b_values = b_values[1:]

    logging.info('b-values linear in q: {}'.format(b_values))

    return b_values


def bvalue_lin_b(bmin = 0.0, bmax = 3000.0, nb_of_b_inside = 2, exclude_bmin = True, verbose = 1):
    """
    Compute b-values linearly distributed in b-value in the 
    interval [bmin, bmax].

    Parameters
    ----------
    bmin: float, Minimum b-value, lower b-value bounds.
    bmax: float, Maximum b-value, upper b-value bounds.
    nb_of_b_inside: integer, number of b-value excluding bmin and bmax.
    exclude_bmin: boolean, exclude bmin from the interval, useful if bmin = 0.0.
    verbose: 0 = silent, 1 = summary upon completion, 2 = print iterations.

    Return
    ------
    b_values: list, increasing b-values.

    """

    b_values = list(np.linspace(bmin, bmax, nb_of_b_inside + 2))
    if exclude_bmin:
        b_values = b_values[1:]

    logging.info('b-values linear in b: {}'.format(b_values))

    return b_values


def add_bvalue_b0(b_values, b0_value = 0.0):
    """
    Add the b0 value to the b-values list.

    Parameters
    ----------
    bvalues: list, b-values of the non-b0 shells.
    b0_value: float, b-values of the b0s
    verbose: 0 = silent, 1 = summary upon completion, 2 = print iterations.

    Return
    ------
    b_values: list, b-values of the shells and b0s.

    """

    b_values.append(b0_value)
    return b_values
