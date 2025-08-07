#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert surface formats

Supported formats:
    ".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"
    and FreeSurfer surfaces

> scil_surface_convert.py surf.vtk converted_surf.ply

Formerly: scil_convert_surface.py
-----------------------------------------------------------------
Reference:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
-----------------------------------------------------------------
"""
import argparse
import logging
import os

from trimeshpy.vtk_util import (load_polydata,
                                save_polydata)

from scilpy.surfaces.utils import (convert_freesurfer_into_polydata,
                                   flip_surfaces_axes)

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_surface',
                   help='Input a surface (FreeSurfer or supported by VTK).')
    p.add_argument('out_surface',
                   help='Output surface (formats supported by VTK).\n'
                        'Recommended extension: .vtk or .ply')

    p.add_argument('--flip_axes', default=[-1, -1, 1], type=int, nargs=3,
                   help='Flip axes for RAS or LPS convention. '
                        'Default is LPS convention (MI-Brain) %(default)s.')
    p.add_argument('--reference',
                   help='Reference image to extract the transformation matrix '
                        'to align the freesurfer surface with the T1.')
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_surface, optional=args.reference)
    assert_outputs_exist(parser, args, args.out_surface)

    _, ext = os.path.splitext(args.in_surface)
    # FreeSurfer surfaces have no extension, verify if the input has one of the
    # many supported extensions, otherwise it is (likely) a FreeSurfer surface
    if ext not in ['.vtk', '.vtp', '.fib', '.ply', '.stl', '.xml', '.obj']:
        if args.reference is None:
            parser.error('The reference image is required for FreeSurfer '
                         'surfaces.')

        polydata = convert_freesurfer_into_polydata(args.in_surface,
                                                    args.reference)
    else:
        polydata = load_polydata(args.in_surface)

    if args.flip_axes:
        polydata = flip_surfaces_axes(polydata, args.flip_axes)

    save_polydata(polydata, args.out_surface, legacy_vtk_format=True)


if __name__ == "__main__":
    main()
