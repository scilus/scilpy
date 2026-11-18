# -*- coding: utf-8 -*-

from dipy.segment.clustering import qbx_and_merge
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from numpy.testing import assert_array_equal

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


def _recognize_shifted_target(shift):
    """
    Tractogram: a target bundle and a distractor bundle 40 mm below it.
    Model: the target bundle, translated by `shift` mm in x.
    BundleSeg is initialized the same way as in VotingScheme.
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
