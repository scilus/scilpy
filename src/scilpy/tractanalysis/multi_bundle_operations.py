# -*- coding: utf-8 -*-
import logging
import time

import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram

from scilpy.image.volume_math import neighborhood_correlation_
from scilpy.tractanalysis.bundle_operations import \
    keep_main_blob_from_bundle_map
from scilpy.tractanalysis.distance_to_centroid import subdivide_bundle
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


def get_correlation_map(sft_list, streamlines_thr=None, correlation_thr=0):
    """
    Get the correlation map of many tractograms representing the same bundle.

    Parameters
    ----------
    sft_list : list[StatefulTractogram]
        The bundles.
    streamlines_thr : float
        Threshold for the minimum number of streamlines in a voxel to be
        included.
    correlation_thr : float
        Threshold for the correlation map.

    Returns
    -------
    corr_map : np.ndarray
        The correlation map
    binary_mask_thresh : np.ndarray
        The mask of included voxels
    binary_mask : np.ndarray
        The binary mask of all voxels before applying the correlation_thr
        (stramlines_thr) is still applied.
    binary_list: list
        The mask for each bundle (before correlation_thr).
    """

    #For each bundle, get the streamline density map and its binary version
    density_list = []
    binary_list = []
    timer = time.time()
    for sft in sft_list:
        density = compute_tract_counts_map(sft.streamlines,
                                           sft.dimensions).astype(float)
        binary = np.zeros(sft.dimensions, dtype=np.uint8)
        if streamlines_thr is not None:
            binary[density >= streamlines_thr] = 1
        else:
            binary[density > 0] = 1
        binary_list.append(binary)
        density_list.append(density)
    logging.debug('Computed density and binary map(s) in '
                  f'{round(time.time() - timer, 3)}.')

    # If we have more than one bundle, get the neighborhood correlation
    if len(density_list) > 1:
        timer = time.time()
        corr_map = neighborhood_correlation_(density_list)
        logging.debug(f'Computed correlation map in '
                      f'{round(time.time() - timer, 3)} seconds')
    else:
        corr_map = density_list[0].astype(float)
        corr_map[corr_map > 0] = 1

    # Creating a single binary mask from all bundles
    binary_mask_nothresh = np.max(binary_list, axis=0)
    binary_mask = binary_mask_nothresh.copy()

    # Apply thresholds if wanted
    if correlation_thr > 1e-3:
        binary_mask[corr_map < correlation_thr] = 0

    return corr_map, binary_mask, binary_mask_nothresh, binary_list
