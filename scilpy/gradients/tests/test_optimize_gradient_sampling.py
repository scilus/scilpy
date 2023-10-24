# -*- coding: utf-8 -*-
import numpy as np

from scilpy.gradients.gen_gradient_sampling import generate_gradient_sampling
from scilpy.gradients.optimize_gradient_sampling import (
    add_b0s_to_bvectors, correct_b0s_philips, compute_bvalue_lin_b,
    compute_bvalue_lin_q, compute_min_duty_cycle_bruteforce,
    compute_peak_power)


def test_swap_sampling_eddy():
    pass


def test_add_b0s_and_correct_b0s():
    nb_samples_per_shell = [4]
    points, idx = generate_gradient_sampling(nb_samples_per_shell, 1)

    new_points, new_shells, nb_new = add_b0s_to_bvectors(
        points, idx, start_b0=True, b0_every=None, finish_b0=False)
    # With 4 b-vectors: we should have [b0 1 2 3 4]
    assert len(new_points) == len(points) + 1

    new_points, new_shells, nb_new = add_b0s_to_bvectors(
        points, idx, start_b0=True, b0_every=3, finish_b0=True)
    # With 4 b-vectors: we should have [b0 1 2 b0 3 4 b0]
    assert len(new_points) == len(points) + 3
    assert nb_new == 3

    new_points, new_shells, nb_new = add_b0s_to_bvectors(
        points, idx, start_b0=True, b0_every=3, finish_b0=False)
    # With 4 b-vectors: we should have [b0 1 2 b0 3 4]
    assert len(new_points) == len(points) + 2

    print(new_points)
    print(new_shells)
    new_points, nb_shells = correct_b0s_philips(new_points, new_shells)
    print(new_points)
    print(new_shells)
    assert False


def test_compute_min_duty_cycle_bruteforce():
    nb_samples_per_shell = [16]
    bvals = [1000]
    points, idx = generate_gradient_sampling(nb_samples_per_shell, 1)

    new_points, new_idx = compute_min_duty_cycle_bruteforce(points, idx, bvals)
    assert len(points) == len(new_points)

    # Verifying that they are the same points; up to a permutation.
    new_points = new_points.tolist()
    points = points.tolist()
    for i in range(16):
        assert points[i] in new_points


def test_compute_peak_power():
    nb_samples_per_shell = [4]
    bvals = [1000]
    points, idx = generate_gradient_sampling(nb_samples_per_shell, 1)

    sqrt_val = np.sqrt(np.array([bvals[idx] for idx in idx]))
    q_scheme = np.abs(points * sqrt_val[:, None])
    power_best = compute_peak_power(q_scheme)

    # toDO. What to test here????


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
