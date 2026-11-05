# -*- coding: utf-8 -*-

from dipy.segment.clustering import qbx_and_merge
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from scilpy.io.streamlines import streamlines_to_memmap
from scilpy.segment.bundleseg import BundleSeg


NB_STREAMLINES = 50


def _make_arc_bundle(rng, center, radius=40., nb_points=40):
    """Half-circle bundle in the xy plane, with small random variations."""
    t = np.linspace(0, np.pi, nb_points)
    arc = np.stack([radius * np.cos(t), radius * np.sin(t),
                    np.zeros(nb_points)], axis=1)

    streamlines = []
    for _ in range(NB_STREAMLINES):
        offset = rng.normal(0, 0.5, 3)
        noise = rng.normal(0, 0.3, (nb_points, 3))
        streamlines.append((arc + center + offset + noise).astype(np.float32))
    return streamlines


def _init_shifted_target(shift):
    """
    Tractogram: a target bundle and a distractor bundle 40 mm below it.
    Model: the target bundle, translated by `shift` mm in x.
    BundleSeg is initialized the same way as in VotingScheme.

    Returns the BundleSeg object, the model, and the temporary directory
    holding the memmaps (to clean up at the end of the test).
    """
    rng = np.random.RandomState(0)
    target = _make_arc_bundle(rng, np.array([0., 0., 0.]))
    distractor = _make_arc_bundle(rng, np.array([0., 0., -40.]))
    tractogram = ArraySequence(target + distractor)
    model = ArraySequence([s + np.array([shift, 0., 0.], dtype=np.float32)
                           for s in target])

    cluster_map = qbx_and_merge(tractogram, [45, 35, 25, 12], nb_pts=12,
                                rng=rng, verbose=False)
    centroids = ArraySequence(cluster_map.centroids)
    clusters_indices = ArraySequence([c.indices
                                      for c in cluster_map.clusters])
    clusters_indices._data = clusters_indices._data.astype(np.uint32)
    tmp_dir, memmap_filenames = streamlines_to_memmap(tractogram, 'float16')

    bsg = BundleSeg(memmap_filenames, np.eye(4), clusters_indices,
                    centroids, rng=rng)
    return bsg, model, tmp_dir


def _recognize_shifted_target(shift):
    bsg, model, tmp_dir = _init_shifted_target(shift)
    indices, _ = bsg.recognize(model, model_clust_thr=4, pruning_thr=8)
    tmp_dir.cleanup()
    return np.sort(indices)


def test_recognize():
    # Small shift: the local SLR barely matters, the target (indices 0 to
    # 49) must be recognized and the distractor (50 to 99) must not.
    indices = _recognize_shifted_target(shift=5)
    assert_array_equal(indices, np.arange(NB_STREAMLINES))


def test_recognize_after_local_slr():
    # Regression test for #1371.
    # The first search space reduction (16 mm) finds the target, 15 mm away,
    # and the local SLR moves the model back onto it. The second reduction
    # (14 mm) must use the registered model: around the model's original
    # position, the target is still 15 mm away and would be missed.
    indices = _recognize_shifted_target(shift=15)
    assert_array_equal(indices, np.arange(NB_STREAMLINES))


def test_reduce_search_space():
    # The model is 15 mm from the target: kept at 16 mm, discarded at 14 mm.
    # The distractor (more than 40 mm away) is never kept.
    bsg, model, tmp_dir = _init_shifted_target(shift=15)
    model_centroids = bsg._cluster_model_bundle(model, 4)

    indices_16, centroids_16 = bsg._reduce_search_space(
        model_centroids, neighbors_reduction_thr=16)
    indices_14, centroids_14 = bsg._reduce_search_space(
        model_centroids, neighbors_reduction_thr=14)
    tmp_dir.cleanup()

    assert_array_equal(np.sort(indices_16), np.arange(NB_STREAMLINES))
    assert len(centroids_16) > 0
    assert indices_14.size == 0
    assert len(centroids_14) == 0


def test_register_model_to_neighb():
    # The model is the target shifted by 15 mm in x: the local SLR must move
    # it back by about -15 mm in x, with almost no rotation or scaling.
    bsg, model, tmp_dir = _init_shifted_target(shift=15)
    model_centroids = bsg._cluster_model_bundle(model, 4)
    _, neighb_centroids = bsg._reduce_search_space(
        model_centroids, neighbors_reduction_thr=16)
    matrix = bsg._register_model_to_neighb(model_centroids,
                                           neighb_centroids)
    tmp_dir.cleanup()

    assert matrix.shape == (4, 4)
    assert_allclose(matrix[:3, 3], [-15, 0, 0], atol=1)
    assert_allclose(matrix[:3, :3], np.eye(3), atol=0.05)