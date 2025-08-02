# -*- coding: utf-8 -*-
import numpy as np

from scilpy.gradients.gen_gradient_sampling import generate_gradient_sampling
from scilpy.gradients.optimize_gradient_sampling import (
    add_b0s_to_bvecs, correct_b0s_philips, compute_bvalue_lin_b,
    compute_bvalue_lin_q, compute_min_duty_cycle_bruteforce,
    compute_peak_power)


def test_swap_sampling_eddy():
    pass


def test_add_b0s_and_correct_b0s():
    nb_samples_per_shell = [4]
    bvecs, idx = generate_gradient_sampling(nb_samples_per_shell, 1)

    new_bvecs, new_shells, nb_new = add_b0s_to_bvecs(
        bvecs, idx, start_b0=True, b0_every=None, finish_b0=False)
    # With 4 b-vectors: we should have [b0 1 2 3 4]
    assert len(new_bvecs) == len(bvecs) + 1

    new_bvecs, new_shells, nb_new = add_b0s_to_bvecs(
        bvecs, idx, start_b0=True, b0_every=3, finish_b0=True)
    # With 4 b-vectors: we should have [b0 1 2 b0 3 4 b0]
    assert len(new_bvecs) == len(bvecs) + 3
    assert nb_new == 3

    new_bvecs, new_shells, nb_new = add_b0s_to_bvecs(
        bvecs, idx, start_b0=True, b0_every=3, finish_b0=False)
    # With 4 b-vectors: we should have [b0 1 2 b0 3 4]
    assert len(new_bvecs) == len(bvecs) + 2
    assert new_shells[0] == -1  # -1 = the "b0 shell".

    new_bvecs, new_shells2 = correct_b0s_philips(new_bvecs, new_shells)
    assert np.array_equal(new_shells2, new_shells)
    # We want to verify that all rows are unique per shell.
    for shell_nb in [-1, 0]:
        shell_bvecs = new_bvecs[new_shells2 == shell_nb, :].tolist()
        for i, b in enumerate(shell_bvecs):
            assert b not in shell_bvecs[i + 1:-1]


def test_compute_min_duty_cycle_bruteforce():
    nb_samples_per_shell = [16]
    bvals = [1000]
    bvecs, idx = generate_gradient_sampling(nb_samples_per_shell, 1)

    new_bvecs, new_idx = compute_min_duty_cycle_bruteforce(bvecs, idx, bvals)
    assert len(bvecs) == len(new_bvecs)

    # Verifying that they are the same bvecs; up to a permutation.
    new_bvecs = new_bvecs.tolist()
    bvecs = bvecs.tolist()
    for i in range(16):
        assert bvecs[i] in new_bvecs


def test_compute_peak_power():
    nb_samples_per_shell = [6]
    bvals = [1000]
    bvecs, idx = generate_gradient_sampling(nb_samples_per_shell, 1)

    # q value is proportional to abs of sqrt of b-value.
    # See code in compute_min_duty_cycle_bruteforce
    sqrt_val = np.sqrt(np.array([bvals[idx] for idx in idx]))
    q_scheme = np.abs(bvecs * sqrt_val[:, None])

    power_best = compute_peak_power(q_scheme, ker_size=6)

    # kern size = nb samples, so should simply compute the max of the sum
    # per axis in the whole sample.
    mean_per_axis = np.sum(q_scheme, axis=0)
    max_total = np.max(mean_per_axis)
    assert max_total == power_best


def test_compute_bvalue_lin_q():
    # Linear distribution between 0 and 100000 with one in the middle
    # **after sqrt.
    # Should be 0, 2500, 10000
    bvals = compute_bvalue_lin_q(
        bmin=0.0, bmax=10000, nb_of_b_inside=1, exclude_bmin=False)
    assert len(bvals) == 3
    assert bvals[1] == (np.sqrt(10000) / 2)**2


def test_compute_bvalue_lin_b():
    # Linear distribution between 0 and 100000 with one in the middle
    # Should be 0, 5000, 10000
    bvals = compute_bvalue_lin_b(
        bmin=0.0, bmax=10000, nb_of_b_inside=1, exclude_bmin=False)
    assert len(bvals) == 3
    assert bvals[1] == 5000
