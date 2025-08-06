# -*- coding: utf-8 -*-
import numpy as np

from scilpy.gradients.gen_gradient_sampling import (generate_gradient_sampling,
                                                    energy_comparison)


def test_generate_gradient_sampling():
    # Note. On a normal computer:
    # Time for 1 shell, 6 samples:   0:00:00.013486
    # Time for 1 shell, 60 samples:  0:00:00.907447
    # Time for 1 shell, 360 samples: 0:02:43.226641. Very slow!!
    nb_samples_per_shell = [6, 60]
    bvecs, shell_idx = generate_gradient_sampling(nb_samples_per_shell, 1)
    assert bvecs.shape[0] == 66
    assert bvecs.shape[1] == 3
    assert len(shell_idx) == 66
    assert sum(shell_idx == 0) == 6
    assert sum(shell_idx == 1) == 60

    # Normalized vectors?
    norm = np.sqrt(np.sum(bvecs**2, axis=1))
    assert np.allclose(norm, 1)


def test_energy_comparison_same():
    bvec1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    e1, e2 = energy_comparison(bvec1, bvec1, 1, [3])
    assert e1 == e2


def test_energy_comparison_diff():
    bvec1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0.9, 0.1]])
    bvec2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    e1, e2 = energy_comparison(bvec1, bvec2, 1, [3])
    assert e1/e2 > 1.25
