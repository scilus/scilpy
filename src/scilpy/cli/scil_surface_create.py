#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a surface with marching cube from a mask or a label image.
The surface will be readable with software like MI-Brain.

Example : use wmparc.a2009s.nii.gz with some aseg.stats indices

scil_surface_create.py out_surface.vtk \\
    --in_labels s1a1/mask/S1-A1_wmparc.a2009s.nii.gz\\
    --list_indices 16:32 --opening 2 --smooth 2 -v
-----------------------------------------------------------------
Reference:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
-----------------------------------------------------------------
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np
import trimeshpy
import trimeshpy.vtk_util as vtk_u
from scipy.ndimage import (binary_closing,
                           binary_dilation,
                           binary_erosion,
                           binary_opening,
                           binary_fill_holes)
import mcubes

from scilpy.image.labels import (get_data_as_labels,
                                 merge_labels_into_mask)
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             ranged_type)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    g1 = p.add_argument_group("Input (Labels or Mask)")
    mxg = g1.add_mutually_exclusive_group(required=True)
    mxg.add_argument('--in_labels',
                     help='Path of the atlas (nii or nii.gz).\n'
                          'You can provide a list of indices with '
                          '--list_indices or create a surface per '
                          'index with --each_index.\n'
                          'If no indices are provided, '
                          'it will merge all indices and converted '
                          'to a binary mask.')
    mxg.add_argument('--in_mask',
                     help='Path of the mask (nii or nii.gz).')
    mxg.add_argument('--in_volume',
                     help='Path of the volume (nii or nii.gz).')

    p.add_argument('out_surface',
                   help='Output surface (.vtk)')

    g2 = p.add_argument_group("Options for labels input")
    mxg2 = g2.add_mutually_exclusive_group()
    mxg2.add_argument('--list_indices', nargs='+',
                      help='List of labels indices to use for the surface.')
    mxg2.add_argument('--each_index', action='store_true',
                      help='Create a surface per index. It will use the '
                           'out_surface basename to create the output files.')

    g3 = p.add_argument_group('Options for volume input')
    g3.add_argument('--value', default=0.5,
                    type=ranged_type(float, 0, None, min_excluded=False),
                    help='Isosurface threshold value used. '
                         'This value is called isovalue in mbcube.\n'
                         'Example: For a binary mask (with 0 and 1), '
                         '0.5 will generate a surface in the middle '
                         'of the transition. [%(default)s]')

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
                   help='Fill holes in the image. [%(default)s]')

    p.add_argument('--vtk2vox', action='store_true',
                   help='Keep output surface in voxel space. [%(default)s]')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [], [args.in_labels,
                                     args.in_mask,
                                     args.in_volume])
    assert_outputs_exist(parser, args, args.out_surface)

    masks = []

    if args.in_labels:
        # Default value for isosurface
        args.value = 0.5
        # Load volume
        img = nib.load(args.in_labels)
        labels_volume = get_data_as_labels(img)

        # Removed indices
        if args.list_indices:
            masks.append(merge_labels_into_mask(labels_volume,
                                                ' '.join(args.list_indices)))
        elif args.each_index:
            indices = np.unique(labels_volume)[1:]
            for index in indices:
                masks.append(merge_labels_into_mask(labels_volume, str(index)))
        else:
            logging.warning('No indices provided, '
                            'it will use all indices.')
            masks.append(labels_volume > 0)
    elif args.in_mask:
        # Default value for isosurface
        args.value = 0.5
        # Load mask
        img = nib.load(args.in_mask)
        masks.append(get_data_as_mask(img))
    else:
        # Load volume
        img = nib.load(args.in_volume)
        masks.append(img.get_fdata(dtype=np.float32))

    for it, mask in enumerate(masks):
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
        mesh = trimeshpy.trimesh_vtk.TriMesh_Vtk(triangles.astype(int),
                                                 vertices)

        # Transformation based on the Nifti affine
        if not args.vtk2vox:
            mesh.set_vertices(vtk_u.vox_to_vtk(mesh.get_vertices(),
                                               img))

        # Smooth
        if args.smooth is not None:
            new_vertices = mesh.laplacian_smooth(1, args.smooth,
                                                 l2_dist_weighted=False,
                                                 area_weighted=False,
                                                 backward_step=True)
            mesh.set_vertices(new_vertices)

        if len(masks) == 1:
            vtk_u.save_polydata(mesh.get_polydata(),
                                args.out_surface,
                                legacy_vtk_format=True)
        else:
            base, ext = os.path.splitext(args.out_surface)
            out_name = args.out_surface.replace(ext,
                                                '_{}'.format(indices[it]) + ext)
            vtk_u.save_polydata(mesh.get_polydata(),
                                out_name,
                                legacy_vtk_format=True)


if __name__ == "__main__":
    main()
