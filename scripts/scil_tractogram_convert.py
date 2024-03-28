#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversion of '.tck', '.trk', '.fib', '.vtk' and 'dpy' files using updated file
format standard. TRK file always needs a reference file, a NIFTI, for
conversion. The FIB file format is in fact a VTK, MITK Diffusion supports it.

Formerly: scil_convert_tractogram.py
"""

import argparse
import logging
import os

from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_bbox_arg, add_overwrite_arg,
                             add_reference_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy')

    p.add_argument('output_name',
                   help='Output filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy')

    add_bbox_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram, args.reference)

    in_extension = os.path.splitext(args.in_tractogram)[1]
    out_extension = os.path.splitext(args.output_name)[1]

    if in_extension == out_extension:
        parser.error('Input and output cannot be of the same file format')

    assert_outputs_exist(parser, args, args.output_name)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    save_tractogram(sft, args.output_name, bbox_valid_check=args.bbox_check)


if __name__ == "__main__":
    main()
