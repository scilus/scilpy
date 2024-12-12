# -*- coding: utf-8 -*-

import numpy as np

from scilpy.tractanalysis.voxel_boundary_intersection import\
    subdivide_streamlines_at_voxel_faces


def mrds_metrics_along_streamlines(sft, mrds_pdds,
                                   metrics, max_theta,
                                   length_weighting):
    """
    Compute mean map for a given fixel-specific metric along streamlines.

    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines needed.
    mrds_pdds : ndarray (X, Y, Z, 3*N_TENSORS)
        MRDS principal diffusion directions of the tensors
    metrics : list of ndarray
        Array of shape (X, Y, Z, N_TENSORS) containing the fixel-specific
        metric of interest.
    max_theta : float
        Maximum angle in degrees between the fiber direction and the
        MRDS principal diffusion direction.
    length_weighting : bool
        If True, will weigh the metric values according to segment lengths.
    """

    mrds_sum, weights = \
        mrds_metric_sums_along_streamlines(sft, mrds_pdds,
                                           metrics, max_theta,
                                           length_weighting)

    all_metric = mrds_sum[0]
    for curr_metric in mrds_sum[1:]:
        all_metric += np.abs(curr_metric)

    non_zeros = np.nonzero(all_metric)
    weights_nz = weights[non_zeros]
    for metric_idx in range(len(metrics)):
        mrds_sum[metric_idx][non_zeros] /= weights_nz

    return mrds_sum


def mrds_metric_sums_along_streamlines(sft, mrds_pdds, metrics,
                                       max_theta, length_weighting):
    """
    Compute a sum map along a bundle for a given fixel-specific metric.

    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines needed.
    mrds_pdds : ndarray (X, Y, Z, 3*N_TENSORS)
        MRDS principal diffusion directions (PDDs) of the tensors
    metrics : list of ndarray (X, Y, Z, N_TENSORS)
        Fixel-specific metrics.
    max_theta : float
        Maximum angle in degrees between the fiber direction and the
        MRDS principal diffusion direction.
    length_weighting : bool
        If True, will weight the metric values according to segment lengths.

    Returns
    -------
    metric_sum_map : np.array
        fixel-specific metrics sum map.
    weight_map : np.array
        Segment lengths.
    """

    sft.to_vox()
    sft.to_corner()

    X, Y, Z = metrics[0].shape[0:3]
    metrics_sum_map = np.zeros((len(metrics), X, Y, Z))
    weight_map = np.zeros(metrics[0].shape[:-1])
    min_cos_theta = np.cos(np.radians(max_theta))

    all_split_streamlines =\
        subdivide_streamlines_at_voxel_faces(sft.streamlines)
    for split_streamlines in all_split_streamlines:
        segments = split_streamlines[1:] - split_streamlines[:-1]
        seg_lengths = np.linalg.norm(segments, axis=1)

        # Remove points where the segment is zero.
        # This removes numpy warnings of division by zero.
        non_zero_lengths = np.nonzero(seg_lengths)[0]
        segments = segments[non_zero_lengths]
        seg_lengths = seg_lengths[non_zero_lengths]

        # Those starting points are used for the segment vox_idx computations
        seg_start = split_streamlines[non_zero_lengths]
        vox_indices = (seg_start + (0.5 * segments)).astype(int)

        normalization_weights = np.ones_like(seg_lengths)
        if length_weighting:
            normalization_weights = seg_lengths

        normalized_seg = np.reshape(segments / seg_lengths[..., None], (-1, 3))

        # Reshape MRDS PDDs
        mrds_pdds = mrds_pdds.reshape(mrds_pdds.shape[0],
                                      mrds_pdds.shape[1],
                                      mrds_pdds.shape[2], -1, 3)

        for vox_idx, seg_dir, norm_weight in zip(vox_indices,
                                                 normalized_seg,
                                                 normalization_weights):
            vox_idx = tuple(vox_idx)

            mrds_peak_dir = mrds_pdds[vox_idx]

            cos_theta = np.abs(np.dot(seg_dir.reshape((-1, 3)),
                                      mrds_peak_dir.T))

            metric_val = [0.0]*len(metrics)
            if (cos_theta > min_cos_theta).any():
                fixel_idx = np.argmax(np.squeeze(cos_theta),
                                      axis=0)  # (n_segs)

                for metric_idx, curr_metric in enumerate(metrics):
                    metric_val[metric_idx] = curr_metric[vox_idx][fixel_idx]

            for metric_idx, curr_metric in enumerate(metrics):
                metrics_sum_map[metric_idx][vox_idx] += metric_val[metric_idx] * norm_weight
                weight_map[vox_idx] += norm_weight

    return metrics_sum_map, weight_map
