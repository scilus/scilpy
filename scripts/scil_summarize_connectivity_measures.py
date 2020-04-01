#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Warning not to mix node_wise_as_list and append_json
"""

import argparse
import json
import os

import bct
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_length_matrix',
                   help='.')
    p.add_argument('in_streamline_count_matrix',
                   help='.')
    p.add_argument('out_json',
                   help='Path of the output json.')

    p.add_argument('--node_wise_as_list', action='store_true',
                   help='Keep the node-wise measures as list.')
    p.add_argument('--output_path_length', nargs=2,
                   metavar='PATH_LENGTH, EDGE_COUNT',
                   help='Save the computed path length and edge count matrix.')
    p.add_argument('--append_json', action='store_true',
                   help='If the file already exists, will append to the '
                        'dictionary.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_length_matrix,
                                 args.in_streamline_count_matrix])

    if not args.append_json:
        assert_outputs_exist(parser, args, args.out_json)
    assert_outputs_exist(parser, args, [], args.output_path_length)

    if args.append_json and args.node_wise_as_list:
        parser.error('Cannot use the append option at the same time as listing '
                     'the node values.')

    sc_matrix = load_matrix_in_any_format(args.in_streamline_count_matrix)
    len_matrix = load_matrix_in_any_format(args.in_length_matrix)

    gtm_dict = {}

    gtm_dict['centrality'] = bct.betweenness_wei(len_matrix).tolist()
    ci, gtm_dict['modularity'] = bct.modularity_finetune_und(sc_matrix)
    gtm_dict['assortativity'] = bct.assortativity_wei(sc_matrix, flag=0)
    gtm_dict['participation'] = bct.participation_coef(sc_matrix, ci).tolist()
    gtm_dict['clustering'] = bct.clustering_coef_wu(sc_matrix).tolist()
    gtm_dict['rich_club'] = bct.rich_club_wu(sc_matrix).tolist()
    gtm_dict['degree'] = bct.degrees_und(sc_matrix).tolist()
    gtm_dict['nodal_strength'] = bct.strengths_und(sc_matrix).tolist()
    gtm_dict['local_efficiency'] = bct.efficiency_wei(len_matrix,
                                                      local=True).tolist()
    gtm_dict['global_efficiency'] = bct.efficiency_wei(len_matrix)
    gtm_dict['density'], _, _ = bct.density_und(sc_matrix)

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
        gtm_dict['path_length'] = np.insert(
            gtm_dict['path_length'],
            empty_connections,
            -1, axis=1)
        gtm_dict['edge_count'] = np.insert(
            gtm_dict['edge_count'],
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
        gtm_dict['centrality'] = float(np.average(gtm_dict['centrality']))
        gtm_dict['participation'] = float(
            np.average(gtm_dict['participation']))
        gtm_dict['clustering'] = float(np.average(gtm_dict['clustering']))
        gtm_dict['rich_club'] = float(np.average(gtm_dict['rich_club']))
        gtm_dict['degree'] = float(np.average(gtm_dict['degree']))
        gtm_dict['nodal_strength'] = float(
            np.average(gtm_dict['nodal_strength']))
        gtm_dict['local_efficiency'] = float(
            np.average(gtm_dict['local_efficiency']))

        valid_values_pl = gtm_dict['path_length'][gtm_dict['path_length'] > 0]
        gtm_dict['path_length'] = float(
            np.average(valid_values_pl))

        valid_values_plec = gtm_dict['edge_count'][gtm_dict['edge_count'] > 0]
        gtm_dict['edge_count'] = float(
            np.average(valid_values_plec))
    else:
        gtm_dict['path_length'] = np.average(
            gtm_dict['path_length'], axis=0).tolist()
        gtm_dict['edge_count'] = np.average(
            gtm_dict['edge_count'], axis=0).tolist()

    if os.path.isfile(args.out_json) and not args.overwrite:
        with open(args.out_json) as json_data:
            out_dict = json.load(json_data)
        for key in out_dict.keys():
            if isinstance(out_dict[key], list):
                out_dict[key].append(gtm_dict[key])
            else:
                out_dict[key] = [out_dict[key], gtm_dict[key]]
    else:
        out_dict = gtm_dict

    with open(args.out_json, 'w') as outfile:
        json.dump(out_dict, outfile, indent=1)


if __name__ == "__main__":
    main()
