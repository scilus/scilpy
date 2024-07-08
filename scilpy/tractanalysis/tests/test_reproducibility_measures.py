# -*- coding: utf-8 -*-
import logging
import os

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import load_tractogram
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.reproducibility_measures import \
    tractogram_pairwise_comparison

fetch_data(get_testing_files_dict(), keys=['bst.zip'])


def test_tractogram_pairwise_comparison():
    logging.getLogger().setLevel('DEBUG')

    sft_path = os.path.join(SCILPY_HOME, 'bst', 'template', 'rpt_m.trk')
    sft = load_tractogram(sft_path, 'same')
    sft.to_vox()
    sft.to_corner()

    mask = compute_tract_counts_map(sft.streamlines, sft.dimensions)
    mask[mask > 0] = 1

    sft_1 = StatefulTractogram.from_sft(sft.streamlines[0:100], sft)

    # -----------
    # Test 1) Bundles with nothing similar: density maps will not overlap.
    # -----------
    logging.info("Test 1: No overlap!")
    fake_str = np.asarray([[1, 1, 1],
                           [1.5, 1.5, 1.5],
                           [2, 2, 2]])
    sft_2 = StatefulTractogram.from_sft([fake_str], sft)
    acc_norm, corr_norm, diff_norm, heatmap, out_mask = (
        tractogram_pairwise_comparison(sft_1, sft_2, mask,
                                       skip_streamlines_distance=True))

    for r in [acc_norm, corr_norm, diff_norm, heatmap]:
        assert np.array_equal(r.shape, sft.dimensions)

    # Make sure we had no overlapping voxels
    assert np.count_nonzero(out_mask) == 0

    # -----------
    # Test 2) Full test. Making sure that we have at least 1 overlapping
    # streamline.
    # -----------
    logging.info("Test 2: Overlap!")
    sft_2 = StatefulTractogram.from_sft(sft.streamlines[100:200], sft)
    acc_norm, corr_norm, diff_norm, heatmap, out_mask = (
        tractogram_pairwise_comparison(sft_1, sft_2, mask,
                                       skip_streamlines_distance=False))

    for r in [acc_norm, corr_norm, diff_norm, heatmap]:
        assert np.array_equal(r.shape, sft.dimensions)

    # Make sure we had overlapping voxels
    assert np.count_nonzero(out_mask) == 1077

    # Comparing with values obtained when creating this test.
    np.testing.assert_almost_equal(np.mean(acc_norm[~np.isnan(acc_norm)]),
                                   0.6590763379712203, decimal=6)
    np.testing.assert_almost_equal(np.mean(corr_norm[~np.isnan(corr_norm)]),
                                   0.6263207793235779, decimal=6)
    np.testing.assert_almost_equal(np.max(corr_norm[~np.isnan(corr_norm)]),
                                   0.99676438850212097, decimal=6)
    np.testing.assert_almost_equal(np.mean(diff_norm[~np.isnan(diff_norm)]),
                                   0.7345049471266359, decimal=6)
    np.testing.assert_almost_equal(np.mean(heatmap[~np.isnan(heatmap)]),
                                   0.7395923591441349, decimal=6)

    # Supervise the number of NaNs in each output.
    # Note. Not the same because:
    # - diff looks at streamlines (with a small radius around the voxel),
    # - TODI looks at the segment of streamlines in the voxels,
    # - Correlation works patch-wise,
    # - Heatmap combines all three.
    # So they all have different regional interaction, which leads to NaN being
    # different
    assert np.count_nonzero(np.isnan(acc_norm)) == 877513
    assert np.count_nonzero(np.isnan(corr_norm)) == 877003
    assert np.count_nonzero(np.isnan(diff_norm)) == 877024
    assert np.count_nonzero(np.isnan(heatmap)) == 877598
