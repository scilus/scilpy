# -*- coding: utf-8 -*-

from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np
from scilpy.tractanalysis.grid_intersections import grid_intersections


def lobe_specific_metric_map_along_streamlines(sft, mrds_pdds,
                                               metric, max_theta,
                                               length_weighting):
    """
    Compute mean map for a given lobe-specific metric along streamlines.

    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines needed.
    mrds_pdds : ndarray (X, Y, Z, 3*N_TENSORS)
        MRDS principal diffusion directions of the tensors
    metric : ndarray
        Array of shape (X, Y, Z, N_TENSORS) containing the lobe-specific
        metric of interest.
    max_theta : float
        Maximum angle in degrees between the fiber direction and the
        MRDS principal diffusion direction.
    length_weighting : bool
        If True, will weigh the metric values according to segment lengths.
    """

    fd_sum, weights = \
        lobe_metric_sum_along_streamlines(sft, mrds_pdds,
                                          metric, max_theta,
                                          length_weighting)

    non_zeros = np.nonzero(fd_sum)
    weights_nz = weights[non_zeros]
    fd_sum[non_zeros] /= weights_nz

    return fd_sum


def lobe_metric_sum_along_streamlines(sft, mrds_pdds, metric,
                                      max_theta, length_weighting):
    """
    Compute a sum map along a bundle for a given lobe-specific metric.

    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines needed.
    mrds_pdds : ndarray (X, Y, Z, 3*N_TENSORS)
        MRDS principal diffusion directions (PDDs) of the tensors
    metric : ndarray (X, Y, Z, N_TENSORS)
        The lobe-specific metric of interest. It can be Axial Diffusivity (AD),
        Radial Diffusivity (RD), Fractional Anisotropy (FA) or Mean Diffusivity (MD).
    max_theta : float
        Maximum angle in degrees between the fiber direction and the
        MRDS principal diffusion direction.
    length_weighting : bool
        If True, will weight the metric values according to segment lengths.

    Returns
    -------
    metric_sum_map : np.array
        Lobe-specific metric sum map.
    weight_map : np.array
        Segment lengths.
    """

    sft.to_vox()
    sft.to_corner()

    metric_sum_map = np.zeros(metric.shape[:-1])
    weight_map = np.zeros(metric.shape[:-1])
    min_cos_theta = np.cos(np.radians(max_theta))

    all_crossed_indices = grid_intersections(sft.streamlines)
    for crossed_indices in all_crossed_indices:
        segments = crossed_indices[1:] - crossed_indices[:-1]
        seg_lengths = np.linalg.norm(segments, axis=1)

        # Remove points where the segment is zero.
        # This removes numpy warnings of division by zero.
        non_zero_lengths = np.nonzero(seg_lengths)[0]
        segments = segments[non_zero_lengths]
        seg_lengths = seg_lengths[non_zero_lengths]

        # Those starting points are used for the segment vox_idx computations
        seg_start = crossed_indices[non_zero_lengths]
        vox_indices = (seg_start + (0.5 * segments)).astype(int)

        normalization_weights = np.ones_like(seg_lengths)
        if length_weighting:
            normalization_weights = seg_lengths

        normalized_seg = np.reshape(segments / seg_lengths[..., None], (-1, 3))

        # Reshape MRDS PDDs
        mrds_pdds = mrds_pdds.reshape(mrds_pdds.shape[0],mrds_pdds.shape[1],mrds_pdds.shape[2],-1,3)

        for vox_idx, seg_dir, norm_weight in zip(vox_indices,
                                                 normalized_seg,
                                                 normalization_weights):
            vox_idx = tuple(vox_idx)

            mrds_peak_dir = mrds_pdds[vox_idx]

            cos_theta = np.abs(np.dot(seg_dir.reshape((-1, 3)),
                                      mrds_peak_dir.T))

            metric_val = 0.0
            if (cos_theta > min_cos_theta).any():
                lobe_idx = np.argmax(np.squeeze(cos_theta), axis=0)  # (n_segs)
                metric_val = metric[vox_idx][lobe_idx]

            metric_sum_map[vox_idx] += metric_val * norm_weight
            weight_map[vox_idx] += norm_weight

    return metric_sum_map, weight_map
