#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate graph theory measures from structural connectivity matrices from
outputs of diffusion MRI tractography. A length-weighted matrix is
optional but required to extract measures such as: global efficiency,
local efficiency, betweeness centrality, path length, edge count, and
small-world omega/sigma. These are not computed if a length matrix is
not provided. The other computed connectivity measures that do
not require the length matrix are: modularity, assortativity,
participation, clustering, nodal_strength, and rich_club.

This script evaluates the measures one subject at the time. To generate a
population dictionary (similarly to other scil_connectivity_* scripts), use
the --append_json option as well as using the same output filename.
>>> for i in hcp/*/; do scil_connectivity_structural_graph_measures
    ${i}/sc_prob.npy ${i}/len_prob.npy hcp_prob.json
    --append_json --avg_node_wise; done

Some measures output one value per node, the default behavior is to list
them all. To obtain only the average use the --avg_node_wise option.

For more details about the measures, please refer to
- https://sites.google.com/site/bctnet/
- https://github.com/aestrivex/bctpy/wiki

This script is under the GNU GPLv3 license, for more detail please refer to
https://www.gnu.org/licenses/gpl-3.0.en.html

----------------------------------------------------------------------------
Reference:
[1] Rubinov, Mikail, and Olaf Sporns. "Complex network measures of brain
    connectivity: uses and interpretations." Neuroimage 52.3 (2010):
    1059-1069.
----------------------------------------------------------------------------
"""

import argparse
import json
import logging
import os

from scilpy.connectivity.matrix_tools import evaluate_graph_measures
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_conn_matrix',
                   help='Input structural connectivity matrix (.npy).')
    p.add_argument('out_json',
                   help='Path of the output json.')

    p.add_argument('--length_matrix',
                   help='Input length-weighted matrix (.npy).')
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


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_conn_matrix)

    if not args.append_json:
        assert_outputs_exist(parser, args, args.out_json)
    else:
        logging.info('Using --append_json, make sure to delete {} '
                     'before re-launching a group analysis.'.format(
                         args.out_json))

    if args.append_json and args.overwrite:
        parser.error('Cannot use the append option at the same time as '
                     'overwrite.\nAmbiguous behavior, consider deleting the '
                     'output json file first instead.')

    conn_matrix = load_matrix_in_any_format(args.in_conn_matrix)
    len_matrix = None

    if args.length_matrix:
        len_matrix = load_matrix_in_any_format(args.length_matrix)

    if args.filtering_mask:
        mask_matrix = load_matrix_in_any_format(
            args.filtering_mask).astype(bool)
        conn_matrix *= mask_matrix

        if args.length_matrix:
            len_matrix *= mask_matrix

    if len_matrix is None:
        print("Warning: No length-weighted matrix provided. ")

    gtm_dict = evaluate_graph_measures(conn_matrix, len_matrix,
                                       args.avg_node_wise, args.small_world)

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
