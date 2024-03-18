#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correct B1 map header problem, by applying the scaling (slope) and setting
the slope to 1.
"""

import argparse
import json
import logging

import nibabel as nib

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.reconst.mti import (adjust_B1_map_header)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_B1_map',
                   help='Path to input B1 map file.')
    p.add_argument('out_B1_map',
                   help='Path to output B1 map file.')
    p.add_argument('in_B1_json',
                   help='Json file of the B1 map.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_outputs_exist(parser, args, args.out_B1_map)
    assert_inputs_exist(parser, [args.in_B1_map, args.in_B1_json])

    with open(args.in_B1_json) as curr_json:
        b1_json = json.load(curr_json)
        if 'PhilipsRescaleSlope' in b1_json.keys():
            slope = b1_json['PhilipsRescaleSlope']
        else:
            raise ValueError('Rescale slope not in Json file.')

    b1_img = nib.load(args.in_B1_map)

    new_b1_img = adjust_B1_map_header(b1_img, slope)

    nib.save(new_b1_img, args.out_B1_map)


if __name__ == '__main__':
    main()
