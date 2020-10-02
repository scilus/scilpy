#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import argparse
import logging
import os

from dipy.io.utils import is_header_compatible
import nibabel as nib
from nibabel.streamlines import Tractogram
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


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractograms)
    assert_outputs_exist(parser, args, args.out_tractogram)

    header = None
    out_streamlines = []
    for in_file in args.in_tractograms:
        _, ext = os.path.splitext(in_file)
        if ext == '.trk':
            if header is None:
                header = nib.streamlines.load(in_file, lazy_load=True).header
            elif not is_header_compatible(header, in_file):
                logging.warning('Incompatible headers in the list.')

        tractogram_file = nib.streamlines.load(in_file)
        out_streamlines.extend(tractogram_file.streamlines)
    out_tractogram = Tractogram(out_streamlines,
                                affine_to_rasmm=np.eye(4))
    nib.streamlines.save(out_tractogram, args.out_tractogram, header=header)


if __name__ == "__main__":
    main()
