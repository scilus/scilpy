# -*- coding: utf-8 -*-
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scipy.sparse import dok_matrix

from scilpy.tractograms.tractogram_operations import union_robust, \
    intersection_robust


def filter_by_occurrence(sft_list, vol_dim, ratio_voxels = 0.5,
                         ratio_streamlines = 0.5):
    """
    Use a list of bundles (representing the same bundle in a population, for
    instance) to keep the voxels representing the average population bundle.

    Parameters
    ----------
    sft_list : list[StatefulTractogram]
        The bundles.
    vol_dim : int, int, int
        The dimension of the volume.
    ratio_voxels : float, optional
        The threshold on the ratio of bundles that must contain a voxel for it
        to be kept.
    ratio_streamlines : float, optional
        The threshold on the ratio of bundles that must contain a streamline
        for it to be kept.

    Returns
    -------
    volume: np.ndarray
        The volume representing the average bundle.
    new_sft: StatefulTractogram
        The tractogram reprensenting the average bundle.
    """
    nb_bundles = len(sft_list)

    fusion_streamlines = []
    for sft in sft_list:
        fusion_streamlines.append(sft.streamlines)

    fusion_streamlines, _ = union_robust(fusion_streamlines)

    volume = np.zeros(vol_dim)
    streamlines_vote = dok_matrix((len(fusion_streamlines), nb_bundles))

    for i in range(nb_bundles):
        # Add an occurrence to voxels touched by this bundle.
        binary = compute_tract_counts_map(sft_list[i].streamlines, vol_dim)
        volume[binary > 0] += 1

        # Remember streamlines in this bundle from the fusion streamlines
        if ratio_streamlines is not None:
            _, indices = intersection_robust([fusion_streamlines,
                                              sft_list[i].streamlines])
            streamlines_vote[list(indices), [i]] += 1

    # Create a tractogram with streamlines well represented
    new_sft = None
    if ratio_streamlines is not None:
        ratio_value = int(ratio_streamlines * nb_bundles)
        real_indices = np.where(np.sum(streamlines_vote,
                                       axis=1) >= ratio_value)[0]
        new_sft = StatefulTractogram.from_sft(fusion_streamlines[real_indices],
                                              sft_list[0])

    # Create a volume with voxels well represented
    volume[volume < int(ratio_voxels * nb_bundles)] = 0
    volume[volume > 0] = 1

    return volume, new_sft
