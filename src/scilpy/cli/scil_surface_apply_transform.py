#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to apply a transform to a surface (FreeSurfer or VTK supported),
using output from ANTs registration tools (i.e. vtk_transfo.txt, output1InverseWarp.nii.gz).

Example usage from T1 to b0 using ANTs transforms:
> ConvertTransformFile 3 output0GenericAffine.mat vtk_transfo.txt --hm
> scil_surface_apply_transform.py lh_white_lps.vtk vtk_transfo.txt lh_white_b0.vtk\\
    --in_deformation output1InverseWarp.nii.gz --inverse

Important: The input surface needs to be in *T1 world LPS* coordinates
(aligned over the T1 in MI-Brain).

The script will use the linear affine first and then the warp image.
The resulting surface will be in *b0 world LPS* coordinates
(aligned over the b0 in MI-Brain).

Formerly: scil_apply_transform_to_surface.py.
-------------------------------------------------------------------
Reference:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
-------------------------------------------------------------------
"""

import argparse
import logging

import nibabel as nib
from trimeshpy.io import load_mesh_from_file
from trimeshpy.vtk_util import save_polydata

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)
from scilpy.surfaces.surface_operations import apply_transform
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_moving_surface',
                   help='Input surface (.vtk).')
    p.add_argument('in_transfo',
                   help='Path of the file containing the 4x4 \n'
                        'transformation, matrix (.txt, .npy or .mat).')
    p.add_argument('out_surface',
                   help='Output surface (.vtk).')

    g = p.add_argument_group("Transformation options")
    g.add_argument('--inverse', action='store_true',
                   help='Apply the inverse linear transformation.')
    g.add_argument('--in_deformation', metavar='file',
                   help='Path to the file containing a deformation field.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_moving_surface, args.in_transfo],
                        args.in_deformation)
    assert_outputs_exist(parser, args, args.out_surface)

    # Load mesh
    mesh = load_mesh_from_file(args.in_moving_surface)

    # Load transformation
    transfo = load_matrix_in_any_format(args.in_transfo)

    deformation_data = None
    if args.in_deformation is not None:
        deformation_data = nib.load(args.in_deformation)

    mesh = apply_transform(mesh, transfo, deformation_data,
                           inverse=args.inverse)

    # Save mesh
    save_polydata(mesh.__polydata__, args.out_surface, legacy_vtk_format=True)


if __name__ == "__main__":
    main()
