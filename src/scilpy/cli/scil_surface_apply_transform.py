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

from dipy.io.surface import save_surface, load_surface
import nibabel as nib

from scilpy.io.surfaces import load_surface_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_surface_spatial_arg,
                             add_verbose_arg,
                             add_vtk_legacy_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             convert_stateful_str_to_enum,
                             load_matrix_in_any_format)
from scilpy.surfaces.surface_operations import apply_transform
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_moving_surface',
                   help='Input surface (.vtk).')
    p.add_argument('in_target_reference',
                   help='Input target reference surface (.vtk).')
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

    add_vtk_legacy_arg(p)
    add_surface_spatial_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p

import numpy as np
def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_moving_surface, args.in_transfo],
                        args.in_deformation)
    assert_outputs_exist(parser, args, args.out_surface)
    convert_stateful_str_to_enum(args)

    # Load data
    sfs = load_surface(args.in_moving_surface, 'freesurfer_t1.nii.gz',
                       from_space=args.source_space,
                       from_origin=args.source_origin)
    
    img = nib.load(args.in_target_reference)
    transfo = load_matrix_in_any_format(args.in_transfo)

    deformation_data = None
    if args.in_deformation is not None:
        deformation_data = nib.load(args.in_deformation)

    out_sfs = apply_transform(sfs, transfo, img, deformation_data,
                           inverse=args.inverse)

    # Save mesh
    save_surface(out_sfs, args.out_surface,
                 to_space=args.destination_space,
                 to_origin=args.destination_origin,
                 legacy_vtk_format=args.legacy_vtk_format)


if __name__ == "__main__":
    main()
