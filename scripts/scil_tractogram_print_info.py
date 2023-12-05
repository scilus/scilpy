#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prints the information on a loaded tractogram: number of streamlines,
mean lengths of streamlines, mean step size.

For trk files: also prints the data_per_point keys and
data_per_streamline keys.

See also: scil_print_header.py to see the header, affine, volume dimension,
etc.
"""
import argparse

import numpy as np
from dipy.tracking.streamlinespeed import length

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import assert_inputs_exist, add_reference_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram')
    add_reference_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    print("Number of streamlines: ", len(sft))

    lengths = [len(s) for s in sft.streamlines]
    lengths_mm = length(sft.streamlines)

    # \u00B1 is the plus or minus sign.
    print(u"Lengths of streamlines: {:.2f} \u00B1 {:.2f} points"
          .format(np.mean(lengths), np.std(lengths)))
    print(u"Lengths of streamlines: {:.2f} \u00B1 {:.2f} mm"
          .format(np.mean(lengths_mm), np.std(lengths_mm)))

    sft.to_voxmm()
    steps = [np.sqrt(np.sum(np.diff(s, axis=0)**2, axis=1))
             for s in sft.streamlines]
    steps = np.hstack(steps)
    print(u"Step size: {:.2f} \u00B1 {:.2f} mm"
          .format(np.mean(steps), np.std(steps)))

    print("Data per point keys: ", list(sft.data_per_point.keys()))
    print("Data per streamline keys: ", list(sft.data_per_streamline.keys()))



if __name__ == '__main__':
    main()
