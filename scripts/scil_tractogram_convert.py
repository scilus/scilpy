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
from dipy.tracking.streamline import transform_streamlines
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_bbox_arg, add_overwrite_arg,
                             add_reference_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg)
from trimeshpy.vtk_util import lines_to_vtk_polydata, save_polydata
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractogram',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy')

    p.add_argument('output_name',
                   help='Output filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy')

    p.add_argument('--legacy_vtk', action='store_true',
                   help='Use the legacy VTK format for streamlines. '
                        'This is the old VTK format, which is supported '
                        'by MI-Brain.')

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

    if out_extension not in ['.vtk', '.fib'] and args.legacy_vtk:
        parser.error(
            'The legacy VTK format is only available for VTK and FIB files')

    if in_extension == out_extension:
        parser.error('Input and output cannot be of the same file format')

    assert_outputs_exist(parser, args, args.output_name)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    if not args.legacy_vtk:
        save_tractogram(sft, args.output_name,
                        bbox_valid_check=args.bbox_check)
    else:
        # This is outside of StatefulTractogram because it is only useful for
        # QC or debugging. Older VTK format is not supported by Dipy.
        transform = np.eye(4)
        transform[0, 0] = -1
        transform[1, 1] = -1
        sft.streamlines = transform_streamlines(sft.streamlines,
                                                transform)
        polydata = lines_to_vtk_polydata(sft.streamlines)
        save_polydata(polydata, args.output_name, binary=True,
                      legacy_vtk_format=args.legacy_vtk)


if __name__ == "__main__":
    main()
