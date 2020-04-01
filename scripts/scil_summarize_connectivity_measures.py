#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import argparse
import itertools
import json
import logging
import multiprocessing
import os

from dipy.io.streamline import load_tractogram
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)



def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_length_matrix',
                   help='.')
    p.add_argument('in_streamline_count_matrix',
                   help='.')
    p.add_argument('out_json',
                   help='Path of the output json.')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_length_matrix,
                                 args.in_streamline_count_matrix])
    assert_outputs_exist(parser, args, args.out_json)

    gtm_dict = {}

    with open(args.out_json, 'w') as outfile:
        json.dump(gtm_dict, outfile, indent=1)


if __name__ == "__main__":
    main()
