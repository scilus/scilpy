# -*- coding: utf-8 -*-
import numpy as np

from scilpy.reconst.fiber_coherence import (compute_coherence_table_for_transforms,
                                            compute_fiber_coherence)


def test_compute_fiber_coherence_fliptable():
    # Just checking that we get 24 values.
    # See below for the real tests.
    directions = np.zeros((3, 3, 5, 3), dtype=float)
    fa = np.zeros((3, 3, 5), dtype=float)
    coherence, transforms = compute_coherence_table_for_transforms(
        directions, fa)
    assert len(coherence) == 24
    assert len(transforms) == 24


def test_compute_fiber_coherence():
    # Coherence will be strong if we have voxels were the peak points towards
    # the neighbor. Ex: Imagine the corpus callosum, where the voxels in X
    # all have peaks in X.

    # Test 1.
    # Aligned on the last dimension (z), we have 4 peaks all pointing in the
    # z direction, with strong FA.
    # There should be a good coherence (actually we get 10).
    directions = np.zeros((3, 3, 5, 3), dtype=float)
    directions[1, 1, :, :] = np.asarray([[0, 0, 1],
                                         [0, 0, 1],
                                         [0, 0, 1],
                                         [0, 0, 1],
                                         [0, 0, 0]], dtype=float)
    fa = np.zeros((3, 3, 5), dtype=float)
    fa[1, 1, :] = [1, 1, 1, 1, 0]
    coherence1 = compute_fiber_coherence(directions, fa)
    assert coherence1 > 0

    # Test 2.
    # Testing symmetry: reversing the 4th voxel should not change the result.
    directions[1, 1, :, :] = np.asarray([[0, 0, 1],
                                         [0, 0, 1],
                                         [0, 0, 1],
                                         [0, 0, -1],
                                         [0, 0, 0]], dtype=float)
    coherence2 = compute_fiber_coherence(directions, fa)
    assert coherence2 == coherence1

    # Test 3
    # Same directions, but with low FA.
    # There should be a lower coherence (actually we get 2).
    fa = np.zeros((3, 3, 5), dtype=float)
    fa[1, 1, :] = [0.2, 0.2, 0.2, 0.2, 0]
    coherence3 = compute_fiber_coherence(directions, fa)
    assert coherence3 < coherence2

    # Test 4.
    # Voxels with non-zero peaks still have peaks in z, but they are aligned
    # in y. There should be a 0 coherence.
    directions = np.zeros((3, 5, 3, 3), dtype=float)
    directions[1, :, 1, :] = np.asarray([[0, 0, 1],
                                         [0, 0, 1],
                                         [0, 0, 1],
                                         [0, 0, -1],
                                         [0, 0, 0]], dtype=float)
    fa = np.zeros((3, 3, 5), dtype=float)
    fa[1, 1, :] = [1, 1, 1, 1, 0]
    coherence4 = compute_fiber_coherence(directions, fa)
    assert coherence4 == 0
