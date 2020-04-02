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
them all into a single value. To obtain all values as a list use the
--node_wise_as_list option.

The computed connectivity measures are:
centrality, modularity, assortativity, participation, clustering, degree
nodal_strength, local_efficiency, global_efficiency, density, rich_club
path_length, edge_count

For more details about the measures, please refer to
- https://sites.google.com/site/bctnet/measures
- https://github.com/aestrivex/bctpy/wiki
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
                             load_matrix_in_any_format,
                             save_matrix_in_any_format)


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
                   help='Input length weighted matrix.')
    p.add_argument('in_streamline_count_matrix',
                   help='Input streamline count weighted matrix..')
    p.add_argument('out_json',
                   help='Path of the output json.')

    p.add_argument('--output_path_length', nargs=2,
                   metavar='PATH_LENGTH, EDGE_COUNT',
                   help='Save the computed path length and edge count matrix.')

    p.add_argument('--node_wise_as_list', action='store_true',
                   help='Keep the node-wise measures as list.')
    p.add_argument('--append_json', action='store_true',
                   help='If the file already exists, will append to the '
                        'dictionary.')

    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_length_matrix,
                                 args.in_streamline_count_matrix])

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not args.append_json:
        assert_outputs_exist(parser, args, args.out_json)
    else:
        logging.debug('Using --append_json, make sure to delete {} '
                      'before re-launching a group analysis.'.format(
                          args.out_json))
    assert_outputs_exist(parser, args, [], args.output_path_length)

    if args.append_json and args.overwrite:
        parser.error('Cannot use the append option at the same time as '
                     'overwrite.\nAmbiguous behavior, consider deleting the '
                     'output json file first instead')

    sc_matrix = load_matrix_in_any_format(args.in_streamline_count_matrix)
    len_matrix = load_matrix_in_any_format(args.in_length_matrix)

    gtm_dict = {}
    gtm_dict['centrality'] = bct.betweenness_wei(len_matrix).tolist()
    ci, gtm_dict['modularity'] = bct.modularity_finetune_und(sc_matrix, seed=0)
    gtm_dict['assortativity'] = bct.assortativity_wei(sc_matrix, flag=0)
    gtm_dict['participation'] = bct.participation_coef_sign(sc_matrix,
                                                            ci)[0].tolist()
    gtm_dict['clustering'] = bct.clustering_coef_wu(sc_matrix).tolist()
    gtm_dict['degree'] = bct.degrees_und(sc_matrix).tolist()

    if args.node_wise_as_list:
        gtm_dict['module_degree_zscore'] = bct.module_degree_zscore(sc_matrix,
                                                                    ci).tolist()
    else:
        logging.debug('Skipping module_degree_zscore, to obtain this value '
                      'use --node_wise_as_list.')
    gtm_dict['nodal_strength'] = bct.strengths_und(sc_matrix).tolist()
    gtm_dict['local_efficiency'] = bct.efficiency_wei(len_matrix,
                                                      local=True).tolist()
    gtm_dict['global_efficiency'] = bct.efficiency_wei(len_matrix)
    gtm_dict['density'], _, _ = bct.density_und(sc_matrix)

    # Rich club always gives an error for the matrix rank and gives NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tmp_rich_club = bct.rich_club_wu(sc_matrix)
    gtm_dict['rich_club'] = tmp_rich_club[~np.isnan(tmp_rich_club)].tolist()

    # Path length gives an infinite distance for unconnected nodes
    # All of this is simply to fix that
    empty_connections = np.where(np.sum(len_matrix, axis=1) < 0.001)[0]
    if len(empty_connections):
        tmp_len_matrix = np.delete(len_matrix, empty_connections, axis=0)
        tmp_len_matrix = np.delete(tmp_len_matrix, empty_connections, axis=1)
        path_length_tuple = bct.distance_wei(tmp_len_matrix)
    else:
        path_length_tuple = bct.distance_wei(len_matrix)
    gtm_dict['path_length'] = path_length_tuple[0]
    gtm_dict['edge_count'] = path_length_tuple[1]

    if args.node_wise_as_list:
        gtm_dict['path_length'] = np.insert(gtm_dict['path_length'],
                                            empty_connections,
                                            -1, axis=1)
        gtm_dict['edge_count'] = np.insert(gtm_dict['edge_count'],
                                           empty_connections,
                                           -1, axis=1)

        # Path length is a matrix that can be saved
        if args.output_path_length:
            pl_for_saving = np.insert(gtm_dict['path_length'],
                                      empty_connections,
                                      -1, axis=1)
            plec_for_saving = np.insert(gtm_dict['edge_count'],
                                        empty_connections,
                                        -1, axis=0)
            save_matrix_in_any_format(args.output_path_length[0],
                                      pl_for_saving)
            save_matrix_in_any_format(args.output_path_length[1],
                                      plec_for_saving)

    if not args.node_wise_as_list:
        # Shorter and easier to read
        def avg_cast(input):
            return float(np.average(input))

        # Average all node-wise measures into a single value
        gtm_dict['centrality'] = avg_cast(gtm_dict['centrality'])
        gtm_dict['participation'] = avg_cast(gtm_dict['participation'])
        gtm_dict['clustering'] = avg_cast(gtm_dict['clustering'])
        gtm_dict['rich_club'] = avg_cast(gtm_dict['rich_club'])
        gtm_dict['degree'] = avg_cast(gtm_dict['degree'])
        gtm_dict['nodal_strength'] = avg_cast(gtm_dict['nodal_strength'])
        gtm_dict['local_efficiency'] = avg_cast(gtm_dict['local_efficiency'])

        valid_values_pl = gtm_dict['path_length'][gtm_dict['path_length'] > 0]
        gtm_dict['path_length'] = avg_cast(valid_values_pl)
        valid_values_plec = gtm_dict['edge_count'][gtm_dict['edge_count'] > 0]
        gtm_dict['edge_count'] = avg_cast(valid_values_plec)
    else:
        gtm_dict['path_length'] = np.average(gtm_dict['path_length'],
                                             axis=0).tolist()
        gtm_dict['edge_count'] = np.average(gtm_dict['edge_count'],
                                            axis=0).tolist()

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
