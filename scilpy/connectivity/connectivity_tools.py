# -*- coding: utf-8 -*-
import itertools
import warnings
from warnings import simplefilter

import bct
from dipy.utils.optpkg import optional_package
import numpy as np
from scipy.cluster import hierarchy

from scilpy.image.labels import get_data_as_labels
from scilpy.stats.matrix_stats import omega_sigma
from scilpy.tractanalysis.reproducibility_measures import \
    approximate_surface_node

simplefilter("ignore", hierarchy.ClusterWarning)

cl, have_bct, _ = optional_package('bct')


def compute_olo(array):
    """
    Optimal Leaf Ordering permutes a weighted matrix that has a
    symmetric sparsity pattern using hierarchical clustering.

    Parameters
    ----------
    array: ndarray (NxN)
        Connectivity matrix.

    Returns
    -------
    perm: ndarray (N,)
        Output permutations for rows and columns.
    """
    if array.ndim != 2:
        raise ValueError('RCM can only be applied to 2D array.')

    Z = hierarchy.ward(array)
    perm = hierarchy.leaves_list(
        hierarchy.optimal_leaf_ordering(Z, array))

    return perm


def apply_olo(array, perm):
    """
    Apply the permutation from compute_RCM.

    Parameters
    ----------
    array: ndarray (NxN)
        Sparse connectivity matrix.
    perm: ndarray (N,)
        Permutations for rows and columns to be applied.

    Returns
    -------
    ndarray (N,N)
        Reordered array.
    """
    if array.ndim != 2:
        raise ValueError('RCM can only be applied to 2D array.')
    return array[perm].T[perm]


def apply_reordering(array, ordering):
    """
    Apply a non-symmetric array ordering that support non-square output.
    The ordering can contain duplicated or discarded rows/columns.

    Parameters
    ----------
    array: ndarray (NxN)
        Sparse connectivity matrix.
    ordering: list of lists
        First elements of the list is the permutation to apply to the rows.
        First elements of the list is the permutation to apply to the columns.

    Returns
    -------
    tmp_array: ndarray (N,N)
        Reordered array.
    """
    if array.ndim != 2:
        raise ValueError('RCM can only be applied to 2D array.')
    if not isinstance(ordering, list) or len(ordering) != 2:
        raise ValueError('Ordering should be a list of lists.\n'
                         '[[x1, x2,..., xn], [y1, y2,..., yn]]')
    ind_1, ind_2 = ordering
    if (np.array(ind_1) > array.shape[0]).any() \
            or (ind_2 > np.array(array.shape[1])).any():
        raise ValueError('Indices from configuration are larger than the'
                         'matrix size, maybe you need a labels list?')
    tmp_array = array[tuple(ind_1), :]
    tmp_array = tmp_array[:, tuple(ind_2)]

    return tmp_array


def evaluate_graph_measures(conn_matrix, len_matrix, avg_node_wise,
                            small_world):
    """
    toDo Finish docstring

    Parameters
    ----------
    conn_matrix: np.ndarray of shape ??
    len_matrix: np.ndarray of shape ??
    avg_node_wise: bool
        If true, return a single value for node-wise measures.
    small_world: bool
        If true, compute measure related to small worldness (omega and sigma).
        This option is much slower.
    """
    if not have_bct:
        raise RuntimeError("bct ist not installed. Please install to use "
                           "this connectivity script.")
    N = len_matrix.shape[0]

    def avg_cast(_input):
        return float(np.average(_input))

    def list_cast(_input):
        if isinstance(_input, np.ndarray):
            if _input.ndim == 2:
                return np.average(_input, axis=1).astype(np.float32).tolist()
            return _input.astype(np.float32).tolist()
        return float(_input)

    if avg_node_wise:
        func_cast = avg_cast
    else:
        func_cast = list_cast

    gtm_dict = {}
    betweenness_centrality = bct.betweenness_wei(len_matrix) / ((N-1)*(N-2))
    gtm_dict['betweenness_centrality'] = func_cast(betweenness_centrality)
    ci, gtm_dict['modularity'] = bct.modularity_louvain_und(conn_matrix,
                                                            seed=0)

    gtm_dict['assortativity'] = bct.assortativity_wei(conn_matrix,
                                                      flag=0)
    gtm_dict['participation'] = func_cast(bct.participation_coef_sign(
        conn_matrix, ci)[0])
    gtm_dict['clustering'] = func_cast(bct.clustering_coef_wu(conn_matrix))

    gtm_dict['nodal_strength'] = func_cast(bct.strengths_und(conn_matrix))
    gtm_dict['local_efficiency'] = func_cast(bct.efficiency_wei(len_matrix,
                                                                local=True))
    gtm_dict['global_efficiency'] = func_cast(bct.efficiency_wei(len_matrix))
    gtm_dict['density'] = func_cast(bct.density_und(conn_matrix)[0])

    # Rich club always gives an error for the matrix rank and gives NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tmp_rich_club = bct.rich_club_wu(conn_matrix)
    gtm_dict['rich_club'] = func_cast(tmp_rich_club[~np.isnan(tmp_rich_club)])

    # Path length gives an infinite distance for unconnected nodes
    # All of this is simply to fix that
    empty_connections = np.where(np.sum(len_matrix, axis=1) < 0.001)[0]
    if len(empty_connections):
        len_matrix = np.delete(len_matrix, empty_connections, axis=0)
        len_matrix = np.delete(len_matrix, empty_connections, axis=1)

    path_length_tuple = bct.distance_wei(len_matrix)
    gtm_dict['path_length'] = func_cast(path_length_tuple[0])
    gtm_dict['edge_count'] = func_cast(path_length_tuple[1])

    if not avg_node_wise:
        for i in empty_connections:
            gtm_dict['path_length'].insert(i, -1)
            gtm_dict['edge_count'].insert(i, -1)

    if small_world:
        gtm_dict['omega'], gtm_dict['sigma'] = omega_sigma(len_matrix)

    return gtm_dict


def normalize_matrix_from_values(matrix, norm_factor, inverse):
    """
    Parameters
    ----------
    matrix: np.ndarray
        Connectivity matrix
    norm_factor: np.ndarray of shape ?
        Matrix used for edge-wise multiplication. Ex: length or volume of the
        bundles.
    inverse: bool
        If true, divide by the matrix rather than multiply.
    """
    where_above0 = norm_factor > 0
    if inverse:
        matrix[where_above0] /= norm_factor[where_above0]
    else:
        matrix[where_above0] *= norm_factor[where_above0]
    return matrix


def normalize_matrix_from_parcel(matrix, atlas_img, labels_list,
                                 parcel_from_volume):
    """
    Parameters
    ----------
    matrix: np.ndarray
        Connectivity matrix
    atlas_img: nib.Nifti1Image
        Atlas for edge-wise division.
    labels_list: np.ndarray
        The list of labels of interest for edge-wise division.
    parcel_from_volume: bool
        If true, parcel from volume. Else, parcel from surface.
    """
    atlas_data = get_data_as_labels(atlas_img)

    voxels_size = atlas_img.header.get_zooms()[:3]
    if voxels_size[0] != voxels_size[1] \
            or voxels_size[0] != voxels_size[2]:
        raise ValueError('Atlas must have an isotropic resolution.')

    voxels_vol = np.prod(atlas_img.header.get_zooms()[:3])
    voxels_sur = np.prod(atlas_img.header.get_zooms()[:2])

    if len(labels_list) != matrix.shape[0] \
            and len(labels_list) != matrix.shape[1]:
        raise ValueError('labels_list should have the same number of label as '
                         'the input matrix.')

    pos_list = range(len(labels_list))
    all_comb = list(itertools.combinations(pos_list, r=2))
    all_comb.extend(zip(pos_list, pos_list))

    # Prevent useless computations for approximate_surface_node()
    factor_list = []
    for label in labels_list:
        if parcel_from_volume:
            factor_list.append(
                np.count_nonzero(atlas_data == label) * voxels_vol)
        else:
            if np.count_nonzero(atlas_data == label):
                roi = np.zeros(atlas_data.shape)
                roi[atlas_data == label] = 1
                factor_list.append(
                    approximate_surface_node(roi) * voxels_sur)
            else:
                factor_list.append(0)

    for pos_1, pos_2 in all_comb:
        factor = factor_list[pos_1] + factor_list[pos_2]
        if abs(factor) > 0.001:
            matrix[pos_1, pos_2] /= factor
            matrix[pos_2, pos_1] /= factor

    return matrix
