#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to load and transform a surface (FreeSurfer or VTK supported),
This script is using ANTs transform (affine.txt, warp.nii.gz).

Best usage with ANTs from T1 to b0:
> ConvertTransformFile 3 output0GenericAffine.mat vtk_transfo.txt --hm
> scil_transform_surface.py lh_white_lps.vtk affine.txt lh_white_b0.vtk\\
    --ants_warp warp.nii.gz

The input surface needs to be in *T1 world LPS* coordinates
(aligned over the T1 in MI-Brain).
The resulting surface should be aligned *b0 world LPS* coordinates
(aligned over the b0 in MI-Brain).
"""

import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from trimeshpy.io import load_mesh_from_file
import trimeshpy.vtk_util as vtk_u

from scilpy.io.utils import (add_overwrite_arg,
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
                   help='Input surface (.vtk).')

    p.add_argument('ants_affine',
                   help='Affine transform from ANTs (.txt or .mat).')

    p.add_argument('out_surface',
                   help='Output surface (.vtk).')

    p.add_argument('--ants_warp',
                   help='Warp image from ANTs (NIfTI format).')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_surface, args.ants_affine],
                        args.ants_warp)
    assert_outputs_exist(parser, args, args.out_surface)

    # Load mesh
    mesh = load_mesh_from_file(args.in_surface)

    # Affine transformation
    if args.ants_affine:
        # Load affine
        affine = np.loadtxt(args.ants_affine)
        inv_affine = np.linalg.inv(affine)

        # Transform mesh vertices
        mesh.set_vertices(mesh.vertices_affine(inv_affine))

        # Flip triangle face, if needed
        if mesh.is_transformation_flip(inv_affine):
            mesh.set_triangles(mesh.triangles_face_flip())

    if args.ants_warp:
        # Load warp
        warp_img = nib.load(args.ants_warp)
        warp = np.squeeze(warp_img.get_fdata(dtype=np.float32))

        # Get vertices translation in voxel space, from the warp image
        vts_vox = vtk_u.vtk_to_vox(mesh.get_vertices(), warp_img)
        tx = map_coordinates(warp[..., 0], vts_vox.T, order=1)
        ty = map_coordinates(warp[..., 1], vts_vox.T, order=1)
        tz = map_coordinates(warp[..., 2], vts_vox.T, order=1)

        # Apply vertices translation in world coordinates
        mesh.set_vertices(mesh.get_vertices() + np.array([tx, ty, tz]).T)

    # Save mesh
    mesh.save(args.out_surface)


if __name__ == "__main__":
    main()
