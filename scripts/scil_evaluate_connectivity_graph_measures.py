#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate graph theory measures from connectivity matrices.
A length weighted and a streamline count weighted matrix are required since
some measures require one or the other.

This script evaluates the measures one subject at the time. To generate a
population dictionary (similarly to other scil_evaluate_*.py scripts), use the
--append_json option as well as using the same output filename.
>>> for i in hcp/*/; do scil_evaluate_connectivity_measures.py ${i}/sc_prob.npy
    ${i}/len_prob.npy hcp_prob.json --append_json --avg_node_wise; done

Some measures output one value per node, the default behavior is to list
them all into a list. To obtain only the average use the
--avg_node_wise option.

The computed connectivity measures are:
centrality, modularity, assortativity, participation, clustering,
nodal_strength, local_efficiency, global_efficiency, density, rich_club,
path_length, edge_count, omega, sigma

For more details about the measures, please refer to
- https://sites.google.com/site/bctnet/measures
- https://github.com/aestrivex/bctpy/wiki

This script is under the GNU GPLv3 license, for more detail please refer to
https://www.gnu.org/licenses/gpl-3.0.en.html
"""

import argparse
import json
import logging
import os
import warnings

import bct
import numpy as np

from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)

EPILOG = """
[1] Rubinov, Mikail, and Olaf Sporns. "Complex network measures of brain
    connectivity: uses and interpretations." Neuroimage 52.3 (2010):
    1059-1069.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=EPILOG)
    p.add_argument('in_conn_matrix',
                   help='Input connectivity matrix (.npy).\n'
                        'Typically a streamline count weighted matrix.')
    p.add_argument('in_length_matrix',
                   help='Input length weighted matrix (.npy).')
    p.add_argument('out_json',
                   help='Path of the output json.')

    p.add_argument('--filtering_mask',
                   help='Binary filtering mask to apply before computing the '
                        'measures.')
    p.add_argument('--avg_node_wise', action='store_true',
                   help='Return a single value for node-wise measures.')
    p.add_argument('--append_json', action='store_true',
                   help='If the file already exists, will append to the '
                        'dictionary.')
    p.add_argument('--small_world', action='store_true',
                   help='Compute measure related to small worldness (omega '
                        'and sigma).\n This option is much slower.')

    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def omega_sigma(matrix):
    """Returns the small-world coefficients (omega & sigma) of a graph.
    Omega ranges between -1 and 1. Values close to 0 mean the matrix
    features small-world characteristics.
    Values close to -1 mean the network has a lattice structure and values
    close to 1 mean G is a random network.

    A network is commonly classified as small-world if sigma > 1.

    Parameters
    ----------
    matrix : numpy.ndarray
        A weighted undirected graph.
    Returns
    -------
    smallworld : tuple of float
        The small-work coefficients (omega & sigma).
    Notes
    -----
    The implementation is adapted from the algorithm by Telesford et al. [1]_.
    References
    ----------
    .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
           "The Ubiquity of Small-World Networks".
           Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.
           doi:10.1089/brain.2011.0038.
    """
    transitivity_rand_list = []
    transitivity_latt_list = []
    path_length_rand_list = []
    for i in range(10):
        logging.debug('Generating random and lattice matrices, '
                      'iteration #{}.'.format(i))
        random = bct.randmio_und(matrix, 10)[0]
        lattice = bct.latmio_und(matrix, 10)[1]

        transitivity_rand_list.append(bct.transitivity_wu(random))
        transitivity_latt_list.append(bct.transitivity_wu(lattice))
        path_length_rand_list.append(avg_cast(bct.distance_wei(random)[0]))

    transitivity = bct.transitivity_wu(matrix)
    path_length = avg_cast(bct.distance_wei(matrix)[0])
    transitivity_rand = np.mean(transitivity_rand_list)
    transitivity_latt = np.mean(transitivity_latt_list)
    path_length_rand = np.mean(path_length_rand_list)

    omega = (path_length_rand / path_length) - \
        (transitivity / transitivity_latt)
    sigma = (transitivity / transitivity_rand) / \
        (path_length / path_length_rand)

    return float(omega), float(sigma)


def avg_cast(input):
    return float(np.average(input))


def list_cast(input):
    if isinstance(input, np.ndarray):
        if input.ndim == 2:
            return np.average(input, axis=1).astype(np.float32).tolist()
        return input.astype(np.float32).tolist()
    return float(input)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_length_matrix,
                                 args.in_conn_matrix])

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not args.append_json:
        assert_outputs_exist(parser, args, args.out_json)
    else:
        logging.debug('Using --append_json, make sure to delete {} '
                      'before re-launching a group analysis.'.format(
                          args.out_json))

    if args.append_json and args.overwrite:
        parser.error('Cannot use the append option at the same time as '
                     'overwrite.\nAmbiguous behavior, consider deleting the '
                     'output json file first instead.')

    conn_matrix = load_matrix_in_any_format(args.in_conn_matrix)
    len_matrix = load_matrix_in_any_format(args.in_length_matrix)

    if args.filtering_mask:
        mask_matrix = load_matrix_in_any_format(args.filtering_mask)
        conn_matrix *= mask_matrix
        len_matrix *= mask_matrix
    N = len_matrix.shape[0]

    if args.avg_node_wise:
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

    if not args.avg_node_wise:
        for i in empty_connections:
            gtm_dict['path_length'].insert(i, -1)
            gtm_dict['edge_count'].insert(i, -1)

    if args.small_world:
        gtm_dict['omega'], gtm_dict['sigma'] = omega_sigma(len_matrix)

    if os.path.isfile(args.out_json) and args.append_json:
        with open(args.out_json) as json_data:
            out_dict = json.load(json_data)
        for key in gtm_dict.keys():
            if isinstance(out_dict[key], list):
                out_dict[key].append(gtm_dict[key])
            else:
                out_dict[key] = [out_dict[key], gtm_dict[key]]
    else:
        out_dict = {}
        for key in gtm_dict.keys():
            out_dict[key] = [gtm_dict[key]]

    with open(args.out_json, 'w') as outfile:
        json.dump(out_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
