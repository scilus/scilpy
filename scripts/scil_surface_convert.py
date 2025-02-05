#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert surface formats

Supported formats:
    ".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"
    and FreeSurfer surfaces

> scil_surface_convert.py surf.vtk converted_surf.ply

Formerly: scil_convert_surface.py
"""
import argparse
import logging
import os

from dipy.io.surface import load_surface, save_surface

from scilpy.io.utils import (add_vtk_legacy_arg,
                             add_overwrite_arg,
                             add_surface_spatial_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             convert_stateful_str_to_enum)

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_surface',
                   help='Input a surface (FreeSurfer or supported by VTK).')
    p.add_argument('out_surface',
                   help='Output surface (formats supported by VTK).\n'
                        'Recommended extension: .vtk or .ply')
    p.add_argument('--reference',
                   help='Reference image to extract the transformation matrix\n'
                        'to align the freesurfer surface with the T1.')

    r = p.add_mutually_exclusive_group()
    r.add_argument('--ref_pial',
                   help='Reference pial surface to extract the transformation\n'
                   'matrix to align the freesurfer surface with the T1.')
    r.add_argument('--ref_gii',
                   help='Reference gii surface to extract the header '
                   'information to be valid')

    add_vtk_legacy_arg(p)
    add_surface_spatial_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_surface,
                        optional=[args.reference, args.ref_pial, args.ref_gii])
    assert_outputs_exist(parser, args, args.out_surface)
    convert_stateful_str_to_enum(args)

    _, ext = os.path.splitext(args.in_surface)
    # FreeSurfer surfaces have no extension, verify if the input has one of the
    # many supported extensions, otherwise it is (likely) a FreeSurfer surface
    if ext not in ['.vtk', '.vtp', '.fib', '.ply', '.stl', '.xml', '.obj']:
        if args.reference is None:
            parser.error('The reference image is required for FreeSurfer '
                         'surfaces.')

        if args.source_space or args.source_origin:
            print('The source space and source origin can not be changed for '
                  'FreeSurfer surfaces. Will be ignored.')

    _, ext = os.path.splitext(args.out_surface)
    if ext not in ['.vtk', '.vtp', '.fib', '.ply', '.stl', '.xml', '.obj']:
        if args.destination_space or args.destination_origin:
            parser.error('The destination space and destination origin can '
                         'not be changed for FreeSurfer surfaces')

    # Dipy takes care of Freesurfer surfaces (ignore space and origin if any)
    sfs = load_surface(args.in_surface, args.reference,
                       from_space=args.source_space,
                       from_origin=args.source_origin)

    # The ref will be either None or a valid path or both None
    save_surface(sfs, args.out_surface, to_space=args.destination_space,
                 to_origin=args.destination_origin,
                 legacy_vtk_format=args.legacy_vtk_format,
                 ref_pial=args.ref_pial, ref_gii=args.ref_gii)


if __name__ == "__main__":
    main()
