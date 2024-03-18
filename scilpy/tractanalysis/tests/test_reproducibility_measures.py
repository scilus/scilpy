# -*- coding: utf-8 -*-

import os

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import load_tractogram
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.reproducibility_measures import tractogram_pairwise_comparison


def test_tractogram_pairwise_comparison():
    sft_path = os.path.join(SCILPY_HOME, 'bst', 'template', 'rpt_m.trk')
    print(sft_path)
    sft = load_tractogram(sft_path, 'same')
    sft_1 = StatefulTractogram.from_sft(sft.streamlines[0:100], sft)
    sft_2 = StatefulTractogram.from_sft(sft.streamlines[100:200], sft)

    sft.to_vox()
    sft.to_corner()
    mask = compute_tract_counts_map(sft.streamlines, sft.dimensions)
    mask[mask > 0] = 1

    results = tractogram_pairwise_comparison(sft_1, sft_2, mask,
                                             skip_streamlines_distance=False)
    assert len(results) == 4
    for r in results:
        assert np.array_equal(r.shape, sft.dimensions)

    assert np.mean(results[0][~np.isnan(results[0])]) == 0.7171550368952226
    assert np.mean(results[1][~np.isnan(results[1])]) == 0.6063336089511456
    assert np.mean(results[2][~np.isnan(results[2])]) == 0.722988562131705
    assert np.mean(results[3][~np.isnan(results[3])]) == 0.7526672393158469

    assert np.count_nonzero(np.isnan(results[0])) == 877627
    assert np.count_nonzero(np.isnan(results[1])) == 877014
    assert np.count_nonzero(np.isnan(results[2])) == 877034
    assert np.count_nonzero(np.isnan(results[3])) == 877671
