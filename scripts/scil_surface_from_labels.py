#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a surface with marching cube from a mask or a label image.
The surface will be readable with software like MI-Brain.

Example : use wmparc.a2009s.nii.gz with some aseg.stats indices

scil_surface_from_labels.py out_surface.vtk \\
    --in_labels s1a1/mask/S1-A1_wmparc.a2009s.nii.gz\\
    --indices 16:32 --vox2vtk --opening 2 --smooth 2 -v
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

from scilpy.image.labels import get_data_as_labels, merge_labels_into_mask
from scilpy.io.image import get_data_as_mask
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

    g = p.add_argument_group("Input (Labels or Mask)")
    mxg = g.add_mutually_exclusive_group(required=True)
    mxg.add_argument('--in_labels',
                     help='Path of the atlas (nii or nii.gz). '
                     'It will use the indices provided with --indices. '
                     'If no indices are provided, it will use all indices.')
    mxg.add_argument('--in_mask',
                     help='Path of the mask (nii or nii.gz).')

    p.add_argument('out_surface',
                   help='Output surface (.vtk)')

    p.add_argument('--indices', nargs='+',
                   help='List of labels indices to use for the surface.')

    p.add_argument('--value', default=0.5,
                   type=ranged_type(float, 0, None, min_excluded=False),
                   help='Isosurface threshold value used. '
                        'This value is called isovalue '
                        'in mbcube. [%(default)s]')

    morpho_g = p.add_argument_group('Morphology options')
    morpho_g.add_argument('--smooth',
                          type=ranged_type(float, 0, None, min_excluded=False),
                          help='Smoothing size with'
                               ' 1 implicit step. [%(default)s]')
    morpho_g.add_argument('--erosion',
                          type=ranged_type(int, 0, None, min_excluded=False),
                          help='Erosion: number of iterations. [%(default)s]')
    morpho_g.add_argument('--dilation',
                          type=ranged_type(int, 0, None, min_excluded=False),
                          help='Dilation: number of iterations. [%(default)s]')
    morpho_g.add_argument('--opening',
                          type=ranged_type(int, 0, None, min_excluded=False),
                          help='Opening (dilation of the erosion): number '
                               'of iterations. [%(default)s]')
    morpho_g.add_argument('--closing',
                          type=ranged_type(int, 0, None, min_excluded=False),
                          help='Closing (erosion of the dilation): number '
                               'of iterations. [%(default)s]')

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

    assert_inputs_exist(parser, [], [args.in_labels, args.in_mask])
    assert_outputs_exist(parser, args, args.out_surface)

    if args.in_labels:
        # Load volume
        labels_img = nib.load(args.in_labels)
        labels_volume = get_data_as_labels(labels_img)

        # Removed indices
        if args.indices:
            mask = merge_labels_into_mask(labels_volume,
                                          ' '.join(args.indices))
        else:
            logging.warning('No indices provided, '
                            'it will use all indices.')
            mask = merge_labels_into_mask(labels_volume,
                                          "1:" + str(labels_volume.max()))
    else:
        # Load mask
        mask_img = nib.load(args.in_mask)
        mask = get_data_as_mask(mask_img)

    # Basic morphology
    if args.erosion is not None:
        mask = binary_erosion(mask, iterations=args.erosion)
    if args.dilation is not None:
        mask = binary_dilation(mask, iterations=args.dilation)
    if args.opening is not None:
        mask = binary_opening(mask, iterations=args.opening)
    if args.closing is not None:
        mask = binary_closing(mask, iterations=args.closing)

    if args.fill:
        mask = binary_fill_holes(mask)

    # Extract marching cube surface from mask
    vertices, triangles = mcubes.marching_cubes(mask, args.value)

    # Generate mesh
    mesh = trimeshpy.trimesh_vtk.TriMesh_Vtk(triangles.astype(int), vertices)

    # Transformation based on the Nifti affine
    if not args.vox2vtk:
        if args.in_labels:
            mesh.set_vertices(vtk_u.vox_to_vtk(mesh.get_vertices(),
                                               labels_img))
        else:
            mesh.set_vertices(vtk_u.vox_to_vtk(mesh.get_vertices(),
                                               mask_img))

    # Smooth
    if args.smooth > 0:
        new_vertices = mesh.laplacian_smooth(1, args.smooth,
                                             l2_dist_weighted=False,
                                             area_weighted=False,
                                             backward_step=True)
        mesh.set_vertices(new_vertices)

    vtk_u.save_polydata(mesh.get_polydata(),
                        args.out_surface,
                        legacy_vtk_format=True)


if __name__ == "__main__":
    main()
