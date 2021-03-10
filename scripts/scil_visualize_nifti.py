#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 2-dimensional fODF slice loaded from disk.

Given an image of SH coefficients, this script displays a slice in a
given orientation. The user can also add a background on top of which the
fODF are to be displayed. Using a full SH basis, the script can be used to
visualize asymmetric fODF. The user can supply a peaks image to visualize
peaks on top of fODF.
"""

import argparse
import warnings

import nibabel as nib
import numpy as np

from dipy.data import get_sphere, SPHERE_FILES
from dipy.reconst.shm import order_from_ncoef

from scilpy.io.utils import (add_sh_basis_args, add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.io.image import get_data_as_mask
from scilpy.viz.scene_utils import (InteractableSlicer,
                                    create_odf_slicer,
                                    create_texture_slicer,
                                    create_peaks_slicer,
                                    create_scene,
                                    render_scene)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    # SH visualization arguments
    p.add_argument('--sh', default=None, help='SH image file.')

    p.add_argument('--full_basis', action='store_true',
                   help='Use full SH basis to reconstruct SH from '
                        'coefficients.')

    p.add_argument('--sphere', default='symmetric724',
                   choices=sorted(SPHERE_FILES.keys()),
                   help='Name of the sphere used to reconstruct SF. '
                        '[%(default)s]')

    p.add_argument('--sph_subdivide', type=int,
                   help='Number of subdivisions for given sphere. If not '
                        'supplied, use the given sphere as is.')

    p.add_argument('--mask',
                   help='Optional mask file. Only fODF inside '
                        'the mask are displayed.')

    p.add_argument('--colormap', default=None,
                   help='Colormap for the ODF slicer. If None, '
                        'then a RGB colormap will be used. [%(default)s]')

    p.add_argument('--scale', default=0.5, type=float,
                   help='Scaling factor for ODF. [%(default)s]')

    p.add_argument('--radial_scale_off', action='store_true',
                   help='Disable radial scale for ODF slicer.')

    p.add_argument('--norm_off', action='store_true',
                   help='Disable normalization of ODF slicer.')

    add_sh_basis_args(p)

    # Texture image options
    p.add_argument('--texture',
                   help='Texture image file. If RGB, values must '
                        'be between 0 and 255.')

    p.add_argument('--tex_range', nargs=2, metavar=('MIN', 'MAX'), type=float,
                   help='The range of values mapped to range [0, 1] '
                        'for texture image. [(bg.min(), bg.max())]')

    p.add_argument('--tex_opacity', type=float, default=1.0,
                   help='The opacity of the texture image. Opacity of 0.0 '
                        'means transparent and 1.0 is completely visible. '
                        '[%(default)s]')

    p.add_argument('--tex_offset', type=float, default=0.5,
                   help='The offset of the texture image with regard to the'
                        ' other images shown. [%(default)s]')

    p.add_argument('--tex_interpolation',
                   default='nearest', choices={'linear', 'nearest'},
                   help='Interpolation mode for the Texture image. '
                        '[%(default)s]')

    # Peaks input file options
    p.add_argument('--peaks', help='Peaks image file.')

    p.add_argument('--peaks_color', nargs=3, type=float,
                   help='Color used for peaks. if None, '
                        'then a RGB colormap is used. [%(default)s]')

    p.add_argument('--peaks_width', default=1.0, type=float,
                   help='Width of peaks segments. [%(default)s]')

    peaks_scale_group = p.add_mutually_exclusive_group()
    peaks_scale_group.add_argument('--peaks_vals',
                                   help='Peaks values file.')

    peaks_scale_group.add_argument('--peaks_length', default=0.65, type=float,
                                   help='Length of the peaks segments. '
                                        '[%(default)s]')

    # Window configuration options
    p.add_argument('--slice_index', type=int,
                   help='Index of the slice to visualize along a given axis. '
                        'Defaults to middle of volume.')

    p.add_argument('--win_dims', nargs=2, metavar=('WIDTH', 'HEIGHT'),
                   default=(768, 768), type=int,
                   help='The dimensions for the vtk window. [%(default)s]')

    p.add_argument('--vtk_interactor_mode', default='trackball',
                   choices={'image', 'trackball'},
                   help='Specify interactor mode for vtk window. '
                        '[%(default)s]')

    p.add_argument('--orientation', default='axial',
                   choices={'axial', 'coronal', 'sagittal'},
                   help='Name of the axis to visualize. [%(default)s]')

    p.add_argument('--silent', action='store_true',
                   help='Disable interactive visualization.')

    p.add_argument('--output', help='Path to output file.')

    add_overwrite_arg(p)

    return p


def _parse_args(parser):
    args = parser.parse_args()
    inputs = []
    output = []
    if args.sh:
        inputs.append(args.sh)
    if args.texture:
        inputs.append(args.texture)
    if args.peaks:
        if args.full_basis:
            # FURY doesn't support asymmetric peaks visualization
            warnings.warn('Asymmetric peaks visualization is not supported '
                          'by FURY. Peaks shown as symmetric peaks.',
                          UserWarning)
        inputs.append(args.peaks)
        if args.peaks_vals:
            inputs.append(args.peaks_vals)
    else:
        if args.peaks_vals:
            parser.error('Peaks values image supplied without peaks. Specify '
                         'a peaks image with --peaks to use this feature.')
    if len(inputs) == 0:
        parser.error('No input is provided. Please specify at least '
                     'one input to continue.')
    if args.mask:
        inputs.append(args.mask)

    if args.output:
        output.append(args.output)
    else:
        if args.silent:
            parser.error('Silent mode is enabled but no output is specified.'
                         'Specify an output with --output to use silent mode.')

    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, output)

    return args


def validate_order(parser, sh_order, ncoeffs, full_basis):
    """
    Check that the sh order agrees with the number
    of coefficients in the input
    """
    if full_basis:
        expected_ncoeffs = (sh_order + 1)**2
    else:
        expected_ncoeffs = (sh_order + 1) * (sh_order + 2) // 2
    if ncoeffs != expected_ncoeffs:
        parser.error('Invalid number of coefficients for fODF. '
                     'Use --full_basis if your input is in '
                     'full SH basis.')


def validate_mask_matches_volume(parser, volume, mask):
    if mask is None:
        return
    if volume.shape[:3] != mask.shape:
        parser.error('Dimensions mismatch. {0} is not {1}'
                     .format(volume.shape, mask.shape))


def main():
    parser = _build_arg_parser()
    args = _parse_args(parser)

    sph = get_sphere(args.sphere)
    mask = None
    grid_shape = None

    if args.mask:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        grid_shape = mask.shape
    if args.sh:
        sh = nib.load(args.sh).get_fdata(dtype=np.float32)
        sh_order = order_from_ncoef(sh.shape[-1], args.full_basis)
        validate_order(parser, sh_order, sh.shape[-1], args.full_basis)
        validate_mask_matches_volume(parser, sh, mask)
        if grid_shape is None:
            grid_shape = sh.shape[:3]
    if args.texture:
        tex = nib.load(args.texture).get_fdata(dtype=np.float32)
        validate_mask_matches_volume(parser, tex, mask)
        if grid_shape is None:
            grid_shape = tex.shape[:3]
    if args.peaks:
        peaks = nib.load(args.peaks).get_fdata(dtype=np.float32)
        peaks = peaks.reshape(peaks.shape[:3] + (-1, 3))
        validate_mask_matches_volume(parser, peaks, mask)
        if args.peaks_vals:
            peaks_vals = nib.load(args.peaks_vals).get_fdata(dtype=np.float32)
            if peaks_vals.shape[:4] != peaks.shape[:4]:
                raise ValueError('Peaks values dimensions {0} do not agree '
                                 'with peaks dimensions {1}.'.format(
                                     peaks_vals.shape,
                                     peaks.shape))
        if grid_shape is None:
            grid_shape = peaks.shape[:3]

    actors = []
    if args.sh:
        # Instantiate the ODF slicer actor
        odf_actor = create_odf_slicer(sh, mask, sph,
                                      args.sph_subdivide, sh_order,
                                      args.sh_basis, args.full_basis,
                                      args.orientation, args.scale,
                                      not args.radial_scale_off,
                                      not args.norm_off, args.colormap,
                                      args.slice_index)
        actors.append(odf_actor)

    # Instantiate a texture slicer actor if a texture image is supplied
    if args.texture:
        tex_actor = create_texture_slicer(tex, mask,
                                          args.slice_index,
                                          args.tex_range,
                                          args.orientation,
                                          args.tex_opacity,
                                          args.tex_offset,
                                          args.tex_interpolation)
        actors.append(tex_actor)

    # Instantiate a peaks slicer actor if peaks are supplied
    if args.peaks:
        if not args.peaks_vals:
            peaks_vals = np.ones(peaks.shape[:-1]) * args.peaks_length
        peaks_actor = create_peaks_slicer(peaks,
                                          args.orientation,
                                          args.slice_index,
                                          peaks_vals,
                                          mask,
                                          args.peaks_color,
                                          args.peaks_width)
        actors.append(peaks_actor)

    # Prepare and display the scene
    scene = create_scene(actors, args.orientation,
                         args.slice_index, grid_shape)
    interactables = []
    for actor in actors:
        interactables.append(InteractableSlicer(actor, grid_shape,
                                                args.orientation,
                                                args.slice_index))
    render_scene(scene, args.win_dims, args.vtk_interactor_mode,
                 args.output, args.silent, interactables)


if __name__ == '__main__':
    main()
