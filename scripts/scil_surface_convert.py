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

from trimeshpy.vtk_util import (load_polydata,
                                save_polydata)

from scilpy.surfaces.utils import (convert_freesurfer_into_polydata,
                                   extract_xform,
                                   flip_LPS)

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)

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
                   help='Output surface (formats supported by VTK).')

    p.add_argument('--xform',
                   help='Path of the copy-paste output from mri_info \n'
                        'Using: mri_info $input >> log.txt, \n'
                        'The file log.txt would be this parameter')

    p.add_argument('--to_lps', action='store_true',
                   help='Flip for Surface/MI-Brain LPS')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_surface)
    assert_outputs_exist(parser, args, args.out_surface)

    if args.xform:
        with open(args.xform) as f:
            content = f.readlines()
        xform = [x.strip() for x in content]
        xform_matrix = extract_xform(xform)
        xform_translation = xform_matrix[0:3, 3]
    else:
        xform_translation = [0, 0, 0]

    if not ((os.path.splitext(args.in_surface)[1])
            in ['.vtk', '.vtp', '.fib', '.ply', '.stl', '.xml', '.obj']):
        polydata = convert_freesurfer_into_polydata(args.in_surface,
                                                    xform_translation)
    else:
        polydata = load_polydata(args.out_surface)

    if args.to_lps:
        polydata = flip_LPS(polydata)

    save_polydata(polydata, args.out_surface, legacy_vtk_format=True)


if __name__ == "__main__":
    main()
