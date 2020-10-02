#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import argparse
import logging
import os

from dipy.io.utils import is_header_compatible
import nibabel as nib
from nibabel.streamlines import LazyTractogram
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('in_tractograms', metavar='INPUT_FILES', nargs='+',
                   help='The list of input tck/trk.')
    p.add_argument('out_tractogram', metavar='OUTPUT_FILE',
                   help='The output file (tck/trk).')

    add_overwrite_arg(p)

    return p


def list_generator_from_nib(filenames):
    for in_file in filenames:
        tractogram_file = nib.streamlines.load(in_file, lazy_load=True)
        for s in tractogram_file.streamlines:
            yield s


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractograms)
    assert_outputs_exist(parser, args, args.out_tractogram)

    header = None
    for in_file in args.in_tractograms:
        _, ext = os.path.splitext(in_file)
        if ext == '.trk':
            if header is None:
                header = nib.streamlines.load(in_file, lazy_load=True).header
            elif not is_header_compatible(header, in_file):
                logging.warning('Incompatible headers in the list.')

    generator = list_generator_from_nib(args.in_tractograms)
    out_tractogram = LazyTractogram(lambda: generator,
                                    affine_to_rasmm=np.eye(4))
    nib.streamlines.save(out_tractogram, args.out_tractogram, header=header)


if __name__ == "__main__":
    main()
