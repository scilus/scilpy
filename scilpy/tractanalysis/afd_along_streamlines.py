# -*- coding: utf-8 -*-

from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix, sph_harm_ind_list
import numpy as np
from scipy.special import lpn

from scilpy.reconst.utils import find_order_from_nb_coeff
from scilpy.tractanalysis.grid_intersections import grid_intersections


def afd_map_along_streamlines(sft, fodf, fodf_basis, length_weighting,
                              is_legacy=True):
    """
    Compute the mean Apparent Fiber Density (AFD) and mean Radial fODF
    (radfODF) maps along a bundle.

    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines needed.
    fodf : nibabel.image
        fODF with shape (X, Y, Z, #coeffs)
        coeffs depending on the sh_order
    fodf_basis : string
        Has to be descoteaux07 or tournier07
    length_weighting : bool
        If set, will weigh the AFD values according to segment lengths

    Returns
    -------
    afd_sum : np.array
        AFD map (weighted if length_weighting)
    rd_sum : np.array
        rdAFD map (weighted if length_weighting)
    """

    afd_sum, rd_sum, weights = \
        afd_and_rd_sums_along_streamlines(sft, fodf, fodf_basis,
                                          length_weighting,
                                          is_legacy=is_legacy)

    non_zeros = np.nonzero(afd_sum)
    weights_nz = weights[non_zeros]
    afd_sum[non_zeros] /= weights_nz
    rd_sum[non_zeros] /= weights_nz

    return afd_sum, rd_sum


def afd_and_rd_sums_along_streamlines(sft, fodf, fodf_basis,
                                      length_weighting, is_legacy=True):
    """
    Compute the mean Apparent Fiber Density (AFD) and mean Radial fODF (radfODF)
    maps along a bundle.

    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines needed.
    fodf : nibabel.image
        fODF with shape (X, Y, Z, #coeffs).
        #coeffs depend on the sh_order.
    fodf_basis : string
        Has to be descoteaux07 or tournier07.
    length_weighting : bool
        If set, will weigh the AFD values according to segment lengths.
    is_legacy : bool, optional
        Whether or not the SH basis is in its legacy form.

    Returns
    -------
    afd_sum_map : np.array
        AFD map.
    rd_sum_map : np.array
        fdAFD map.
    weight_map : np.array
        Segment lengths.
    """

    sft.to_vox()
    sft.to_corner()

    fodf_data = fodf.get_fdata(dtype=np.float32)
    order = find_order_from_nb_coeff(fodf_data)
    sphere = get_sphere('repulsion724')
    b_matrix, _ = sh_to_sf_matrix(sphere, order, fodf_basis, legacy=is_legacy)
    _, n = sph_harm_ind_list(order)
    legendre0_at_n = lpn(order, 0)[0][n]
    sphere_norm = np.linalg.norm(sphere.vertices)

    afd_sum_map = np.zeros(shape=fodf_data.shape[:-1])
    rd_sum_map = np.zeros(shape=fodf_data.shape[:-1])
    weight_map = np.zeros(shape=fodf_data.shape[:-1])

    p_matrix = np.eye(fodf_data.shape[3]) * legendre0_at_n
    all_crossed_indices = grid_intersections(sft.streamlines)
    for crossed_indices in all_crossed_indices:
        segments = crossed_indices[1:] - crossed_indices[:-1]
        seg_lengths = np.linalg.norm(segments, axis=1)

        # Remove points where the segment is zero.
        # This removes numpy warnings of division by zero.
        non_zero_lengths = np.nonzero(seg_lengths)[0]
        segments = segments[non_zero_lengths]
        seg_lengths = seg_lengths[non_zero_lengths]

        test = np.dot(segments, sphere.vertices.T)
        test2 = (test.T / (seg_lengths * sphere_norm)).T
        angles = np.arccos(test2)
        sorted_angles = np.argsort(angles, axis=1)
        closest_vertex_indices = sorted_angles[:, 0]

        # Those starting points are used for the segment vox_idx computations
        strl_start = crossed_indices[non_zero_lengths]
        vox_indices = (strl_start + (0.5 * segments)).astype(int)

        normalization_weights = np.ones_like(seg_lengths)
        if length_weighting:
            normalization_weights = seg_lengths / \
                np.linalg.norm(fodf.header.get_zooms()[:3])

        for vox_idx, closest_vertex_index, norm_weight in zip(vox_indices,
                                                              closest_vertex_indices,
                                                              normalization_weights):
            vox_idx = tuple(vox_idx)
            b_at_idx = b_matrix.T[closest_vertex_index]
            fodf_at_index = fodf_data[vox_idx]

            afd_val = np.dot(b_at_idx, fodf_at_index)
            rd_val = np.dot(np.dot(b_at_idx.T, p_matrix),
                            fodf_at_index)

            afd_sum_map[vox_idx] += afd_val * norm_weight
            rd_sum_map[vox_idx] += rd_val * norm_weight
            weight_map[vox_idx] += norm_weight

    rd_sum_map[rd_sum_map < 0.] = 0.
    return afd_sum_map, rd_sum_map, weight_map
