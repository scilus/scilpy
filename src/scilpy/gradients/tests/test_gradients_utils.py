# -*- coding: utf-8 -*-
import numpy as np

from scilpy.gradients.bvec_bval_tools import is_normalized_bvecs
from scilpy.gradients.utils import random_uniform_on_sphere, \
    get_new_gtab_order


def test_random_uniform_on_sphere():
    bvecs = random_uniform_on_sphere(10)

    # Confirm that they are unit vectors.
    assert is_normalized_bvecs(bvecs)

    # Can't check much more than that. Supervising that no co-linear vectors.
    # (Each pair of vector should have at least some angle in-between)
    # We also tried to check if the closest neighbor to each vector is more or
    # less always with the same angle, but it is very variable.
    min_expected_angle = 1.0
    smallests = []
    for i in range(10):
        angles = np.rad2deg(np.arccos(np.dot(bvecs[i, :], bvecs.T)))
        # Smallest, except 0 (with itself). Sometimes this is nan.
        smallests.append(np.nanmin(angles[angles > 1e-5]))
    assert np.all(np.asarray(smallests) > min_expected_angle)


def test_get_new_gtab_order():
    # Using N=4 vectors
    philips_table = np.asarray([[1, 1, 1, 1],
                                [2, 2, 2, 2],
                                [3, 3, 3, 3],
                                [4, 4, 4, 4]])
    dwi = np.ones((10, 10, 10, 4))
    bvecs = np.asarray([[3, 3, 3],
                        [4, 4, 4],
                        [2, 2, 2],
                        [1, 1, 1]])
    bvals = np.asarray([3, 4, 2, 1])

    order = get_new_gtab_order(philips_table, dwi, bvals, bvecs)

    assert np.array_equal(bvecs[order, :], philips_table[:, 0:3])
    assert np.array_equal(bvals[order], philips_table[:, 3])
