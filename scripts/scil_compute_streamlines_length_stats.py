#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json

from dipy.tracking.streamlinespeed import length
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_arg,
                             add_reference_arg,
                             assert_inputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute streamlines min, mean and max length, as well as '
                    'standard deviation of length in mm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')

    add_reference_arg(p)
    add_json_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundle)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    streamlines = sft.streamlines
    lengths = [0]
    if streamlines:
        lengths = list(length(streamlines))

    print(json.dumps({'min_length': float(np.min(lengths)),
                      'mean_length': float(np.mean(lengths)),
                      'max_length': float(np.max(lengths)),
                      'std_length': float(np.std(lengths))},
                     indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
