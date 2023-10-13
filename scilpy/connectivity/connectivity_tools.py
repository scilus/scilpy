# -*- coding: utf-8 -*-

import warnings
from warnings import simplefilter

import bct
import numpy as np
from scipy.cluster import hierarchy

from scilpy.stats.matrix_stats import omega_sigma

simplefilter("ignore", hierarchy.ClusterWarning)


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


def parse_ordering(in_ordering_file, labels_list=None):
    with open(in_ordering_file, 'r') as my_file:
        lines = my_file.readlines()
        ordering = [[int(val) for val in lines[0].split()],
                    [int(val) for val in lines[1].split()]]
    if labels_list:
        labels_list = np.loadtxt(labels_list,
                                 dtype=np.int16).tolist()
        # If the reordering file refers to labels and not indices
        real_ordering = [[], []]
        real_ordering[0] = [labels_list.index(i) for i in ordering[0]]
        real_ordering[1] = [labels_list.index(i) for i in ordering[1]]
        return real_ordering

    return ordering


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
    gtm_dict['participation'] = func_cast(bct.participation_coef_sign(conn_matrix,
                                                                      ci)[0])
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
