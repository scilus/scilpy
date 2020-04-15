#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate graph theory measures from connectivity matrices.
A length weighted and a streamline count weighted matrix are required since
some measures require one or the other.

This script evaluate the measure one subject at the time, to generate a
population dictionary (similarly to the other scil_evaluate_*.py ) use the
--append_json option as well as using the same output filename.

Some measures output one value per node, the default behavior is to average
them all into a single value.

The computed connectivity measures are:
degree_centrality, betweenness_centrality, degree_assortativity
clustering, modularity_quality, modularity_coverage, modularity_performance
nodal_strength, local_efficiency, global_efficiency, density, rich_club
path_length, edge_count, omega, sigma

For more details about the measures, please refer to:
- https://sites.google.com/site/bctnet/measures
- https://networkx.github.io/documentation/networkx-1.9/overview.html
"""

import argparse
import json
import logging
import multiprocessing
import os
from time import time
import itertools

import networkx as nx
from networkx.algorithms.assortativity import degree_assortativity_coefficient
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality
from networkx.algorithms.cluster import transitivity
from networkx.algorithms.community import (kernighan_lin_bisection,
                                           modularity, performance, coverage)
from networkx.algorithms.efficiency_measures import (local_efficiency,
                                                     global_efficiency)
from networkx.algorithms.richclub import rich_club_coefficient
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
from networkx.algorithms.smallworld import random_reference, lattice_reference
import numpy as np

from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_processes_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)


EPILOG = """
References:
    [1] Rubinov, Mikail, and Olaf Sporns. "Complex network measures of brain
        connectivity: uses and interpretations." Neuroimage 52.3 (2010):
        1059-1069.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=EPILOG)
    p.add_argument('in_length_matrix',
                   help='Input length weighted matrix (.npy).')
    p.add_argument('in_connection_matrix',
                   help='Input connection matrix (.npy).\n'
                        'Typically a streamline count weighted')
    p.add_argument('out_json',
                   help='Path of the output json.')

    p.add_argument('--filtering_mask',
                   help='Binary filtering mask to apply before computing the '
                        'measures.')

    p.add_argument('--small_world', action='store_true',
                   help='Compute measure related to small worldness (omega '
                   'and sigma).\n This option is much slower.')

    p.add_argument('--append_json', action='store_true',
                   help='If the file already exists, will append to the '
                        'dictionary.')

    add_json_args(p)
    add_verbose_arg(p)
    add_processes_arg(p)
    add_overwrite_arg(p)

    return p


def random_reference_wrapper(args):
    return random_reference(args[0], niter=args[1], seed=args[2])


def lattice_reference_wrapper(args):
    return lattice_reference(args[0], niter=args[1], seed=args[2])


def dict_to_avg(dix):
    array = np.zeros(len(dix))
    for i in dix:
        array[int(i)] = float(dix[i])

    return float(np.average(array))


def omega_sigma(G, nbr_processes, niter=10, nrand=10, seed=None):
    """Returns the small-world coefficient (omega) of a graph
    The small-world coefficient of a graph G is:
    omega = Lr/L - C/Cl
    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Lr is the average shortest path length
    of an equivalent random graph and Cl is the average clustering coefficient
    of an equivalent lattice graph.
    The small-world coefficient (omega) ranges between -1 and 1. Values close
    to 0 means the G features small-world characteristics. Values close to -1
    means G has a lattice shape whereas values close to 1 means G is a random
    graph.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    niter: integer (optional, default=10)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.
    nrand: integer (optional, default=10)
        Number of random graphs generated to compute the average clustering
        coefficient (Cr) and average shortest path length (Lr).
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    Returns
    -------
    omega : float
        The small-work coefficient (omega)
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
    randMetrics = {"C1": [], "L": [], "C2": []}

    pool = multiprocessing.Pool(nbr_processes)
    timer = time()
    Gr_list = pool.map(random_reference_wrapper, zip(itertools.repeat(G),
                                                     itertools.repeat(niter),
                                                     range(nrand)))
    logging.debug('Generated {} random reference matrices in {} sec. using {} '
                  'processes'.format(niter, np.round(time() - timer, 3),
                  nbr_processes))
    Gl_list = pool.map(lattice_reference_wrapper, zip(itertools.repeat(G),
                                                      itertools.repeat(niter),
                                                      range(nrand)))
    pool.close()
    pool.join()

    logging.debug('Generated {} lattice reference matrices in {} sec. using {} '
                  'processes'.format(niter, np.round(time() - timer, 3),
                  nbr_processes))
    pool.close()
    pool.join()

    for i in range(nrand):
        randMetrics["C1"].append(nx.transitivity(Gr_list[i]))
        randMetrics["C2"].append(nx.transitivity(Gl_list[i]))
        randMetrics["L"].append(nx.average_shortest_path_length(Gr_list[i]))

    C = nx.transitivity(G)
    L = nx.average_shortest_path_length(G)
    Cr = np.mean(randMetrics["C1"])
    Cl = np.mean(randMetrics["C2"])
    Lr = np.mean(randMetrics["L"])

    omega = (Lr / L) - (C / Cl)
    sigma = (C / Cr) / (L / Lr)

    return omega, sigma


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_length_matrix,
                                 args.in_connection_matrix])

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
                     'output json file first instead')

    conn_matrix = load_matrix_in_any_format(args.in_connection_matrix)
    conn_matrix /= conn_matrix.max()
    len_matrix = load_matrix_in_any_format(args.in_length_matrix)

    if args.filtering_mask:
        mask_matrix = load_matrix_in_any_format(args.filtering_mask)
        conn_matrix *= mask_matrix
        len_matrix *= mask_matrix

    conn_graph = nx.from_numpy_matrix(conn_matrix)
    len_graph = nx.from_numpy_matrix(len_matrix)

    gtm_dict = {}
    gtm_dict['degree_centrality'] = dict_to_avg(degree_centrality(conn_graph))
    betweenness_centra = betweenness_centrality(len_graph,
                                                weight='weight')
    gtm_dict['betweenness_centrality'] = dict_to_avg(betweenness_centra)
    ci = kernighan_lin_bisection(conn_graph, max_iter=100, seed=0)
    gtm_dict['modularity_quality'] = modularity(conn_graph, ci,
                                                weight='weight')
    gtm_dict['modularity_coverage'] = coverage(conn_graph, ci)
    gtm_dict['modularity_performance'] = performance(conn_graph, ci)
    gtm_dict['assortativity'] = degree_assortativity_coefficient(conn_graph)
    gtm_dict['transitivity'] = transitivity(conn_graph)
    gtm_dict['nodal_strength'] = float(np.average(np.sum(conn_matrix,
                                                         axis=0)))
    gtm_dict['local_efficiency'] = local_efficiency(conn_graph)
    gtm_dict['global_efficiency'] = global_efficiency(conn_graph)

    rich_club = rich_club_coefficient(conn_graph, normalized=False,
                                      seed=0)
    gtm_dict['rich_club'] = dict_to_avg(rich_club)

    # Last measures, does not support the unconnected edges. Removing them.
    empty_connections = np.where(np.sum(len_matrix, axis=1) < 0.001)[0]
    if len(empty_connections):
        tmp_len_matrix = np.delete(len_matrix, empty_connections, axis=0)
        tmp_len_matrix = np.delete(tmp_len_matrix, empty_connections, axis=1)
        len_graph = nx.from_numpy_matrix(tmp_len_matrix)

    gtm_dict['path_length'] = average_shortest_path_length(len_graph,
                                                           weight='weight')
    gtm_dict['edge_count'] = average_shortest_path_length(len_graph)

    if args.small_world:
        gtm_dict['omega'], gtm_dict['sigma'] = omega_sigma(len_graph,
                                                           args.nbr_processes,
                                                           seed=0)

    if os.path.isfile(args.out_json) and args.append_json:
        with open(args.out_json) as json_data:
            out_dict = json.load(json_data)
        for key in gtm_dict.keys():
            out_dict[key].append(gtm_dict[key])
    else:
        out_dict = {}
        for key in gtm_dict.keys():
            out_dict[key] = [gtm_dict[key]]

    with open(args.out_json, 'w') as outfile:
        json.dump(out_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
