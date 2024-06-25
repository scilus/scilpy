#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert segmented volume to a surface with marching cube'

Example : use wmparc.a2009s.nii.gz with some aseg.stats indices

scil_surface_from_volume.py s1a1/mask/S1-A1_wmparc.a2009s.nii.gz\\
    -v -index 16  --vox2vtk -opening 2 -smooth 2

"""

import argparse
import logging

import nibabel as nib
import trimeshpy
import trimeshpy.vtk_util as vtk_u
from scipy.ndimage import (binary_closing,
                           binary_dilation,
                           binary_erosion,
                           binary_opening,
                           binary_fill_holes)
import mcubes

from scilpy.image.labels import get_data_as_labels, get_binary_mask_from_labels
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             ranged_type)

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_label',
                   help='Path of the volume (nii or nii.gz).')

    p.add_argument('out_surface',
                   help='Output surface (.vtk)')

    p.add_argument('--indices', type=int, nargs='+',
                   help='List of labels indices to use for the surface.')

    p.add_argument('--value', default=0.5,
                   type=ranged_type(float, 0, None, min_excluded=False),
                   help='Isosurface threshold value used. '
                        'This value is called isovalue '
                        'in mbcube. [%(default)s]')

    # Need a group here
    morpho_g = p.add_argument_group('Morphology options')
    morpho_g.add_argument('--smooth', default=0.0,
                          type=ranged_type(float, 0, None, min_excluded=False),
                          help='Smoothing size with'
                               ' 1 implicit step. [%(default)s]')
    morpho_g.add_argument('--erosion', default=0,
                          type=ranged_type(int, 0, None, min_excluded=False),
                          help='Erosion: number of iterations. [%(default)s]')
    morpho_g.add_argument('--dilation', default=0,
                          type=ranged_type(int, 0, None, min_excluded=False),
                          help='Dilation: number of iterations. [%(default)s]')
    morpho_g.add_argument('--opening', default=0,
                          type=ranged_type(int, 0, None, min_excluded=False),
                          help='Opening: number of iterations. [%(default)s]')
    morpho_g.add_argument('--closing', default=0,
                          type=ranged_type(int, 0, None, min_excluded=False),
                          help='Closing: number of iterations. [%(default)s]')

    p.add_argument('--fill', action='store_true',
                   help='Fill holes in the image.')

    p.add_argument('--vox2vtk', action='store_true',
                   help='Transformation to vox2vtk.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_label)
    assert_outputs_exist(parser, args, args.out_surface)

    # Load volume
    labels_img = nib.load(args.in_label)
    labels_volume = get_data_as_labels(labels_img)

    # Removed indices
    mask = get_binary_mask_from_labels(labels_volume, args.indices)

    # Basic morphology
    if args.erosion > 0:
        mask = binary_erosion(mask, iterations=args.erosion)
    if args.dilation > 0:
        mask = binary_dilation(mask, iterations=args.dilation)
    if args.opening > 0:
        mask = binary_opening(mask, iterations=args.opening)
    if args.closing > 0:
        mask = binary_closing(mask, iterations=args.closing)

    if args.fill:
        mask = binary_fill_holes(mask)

    # Extract marching cube surface from mask
    vertices, triangles = mcubes.marching_cubes(mask, args.value)

    # Generate mesh
    mesh = trimeshpy.trimesh_vtk.TriMesh_Vtk(triangles.astype(int), vertices)

    # Transformation based on the Nifti affine
    if args.vox2vtk:
        mesh.set_vertices(vtk_u.vox_to_vtk(mesh.get_vertices(), labels_img))

    # Smooth
    if args.smooth > 0:
        new_vertices = mesh.laplacian_smooth(1, args.smooth,
                                             l2_dist_weighted=False,
                                             area_weighted=False,
                                             backward_step=True)
        mesh.set_vertices(new_vertices)

    vtk_u.save_polydata(mesh.get_polydata(), args.out_surface, legacy_vtk_format=True)


if __name__ == "__main__":
    main()
