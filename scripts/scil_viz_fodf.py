#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 2-dimensional fODF slice loaded from disk.

Given an image of SH coefficients, this script displays a slice in a
given orientation. The user can also add a background on top of which the
fODF are to be displayed. Using a full SH basis, the script can be used to
visualize asymmetric fODF. The user can supply a peaks image to visualize
peaks on top of fODF.

If a transparency_mask is given (e.g. a brain mask), all values outside the
mask non-zero values are set to full transparency in the saved scene.

!!! CAUTION !!! The script is memory intensive about (9kB of allocated RAM per
voxel, or 9GB for a 1M voxel volume) with a sphere interpolated to 362 points.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from dipy.data import get_sphere

from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.io.utils import (add_overwrite_arg,
                             add_sh_basis_args,
                             assert_inputs_exist,
                             add_verbose_arg,
                             assert_outputs_exist,
                             parse_sh_basis_arg,
                             assert_headers_compatible)
from scilpy.io.image import assert_same_resolution, get_data_as_mask
from scilpy.utils.spatial import RAS_AXES_NAMES
from scilpy.viz.backends.fury import (create_interactive_window,
                                      create_scene,
                                      snapshot_scenes)
from scilpy.viz.backends.pil import any2grayscale
from scilpy.viz.screenshot import compose_image
from scilpy.viz.slice import (create_odf_slicer,
                              create_peaks_slicer,
                              create_texture_slicer)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    # Positional arguments
    p.add_argument('in_fodf', default=None, help='Input SH image file.')

    # Window configuration options
    p.add_argument('--slice_index', type=int,
                   help='Index of the slice to visualize along a given axis. '
                        'Defaults to middle of volume.')

    p.add_argument('--win_dims', nargs=2, metavar=('WIDTH', 'HEIGHT'),
                   default=(768, 768), type=int,
                   help='The dimensions for the vtk window. [%(default)s]')

    p.add_argument('--interactor', default='trackball',
                   choices={'image', 'trackball'},
                   help='Specify interactor mode for vtk window. '
                        '[%(default)s]')

    p.add_argument('--axis_name', default='axial', type=str,
                   choices=RAS_AXES_NAMES,
                   help='Name of the axis to visualize. [%(default)s]')

    p.add_argument('--silent', action='store_true',
                   help='Disable interactive visualization.')

    p.add_argument('--in_transparency_mask', help='Input mask image file.')

    p.add_argument('--output', help='Path to output file.')

    add_overwrite_arg(p)

    # Optional FODF personalization arguments
    add_sh_basis_args(p)

    sphere_choices = {'symmetric362', 'symmetric642', 'symmetric724',
                      'repulsion724', 'repulsion100', 'repulsion200'}
    p.add_argument('--sphere', default='symmetric362', choices=sphere_choices,
                   help='Name of the sphere used to reconstruct SF. '
                        '[%(default)s]')

    p.add_argument('--sph_subdivide', type=int,
                   help='Number of subdivisions for given sphere. If not '
                        'supplied, use the given sphere as is.')

    p.add_argument('--mask',
                   help='Optional mask file. Only fODF inside '
                        'the mask are displayed.')

    q = p.add_mutually_exclusive_group()

    q.add_argument('--colormap', default=None,
                   help='Colormap for the ODF slicer. If None, '
                        'then a RGB colormap will be used. [%(default)s]')

    q.add_argument('--color_rgb', nargs=3, type=float, default=None,
                   help='Uniform color for the ODF slicer given as RGB, '
                        'scaled between 0 and 1. [%(default)s]')

    p.add_argument('--scale', default=0.5, type=float,
                   help='Scaling factor for FODF. [%(default)s]')

    p.add_argument('--radial_scale_off', action='store_true',
                   help='Disable radial scale for ODF slicer.')

    p.add_argument('--norm_off', action='store_true',
                   help='Disable normalization of ODF slicer.')

    add_verbose_arg(p)

    # Background image options
    bg = p.add_argument_group('Background arguments')
    bg.add_argument('--background',
                    help='Background image file. If RGB, values must '
                         'be between 0 and 255.')

    bg.add_argument('--bg_range', nargs=2, metavar=('MIN', 'MAX'), type=float,
                    help='The range of values mapped to range [0, 1] '
                         'for background image. [(bg.min(), bg.max())]')

    bg.add_argument('--bg_opacity', type=float, default=1.0,
                    help='The opacity of the background image. Opacity of 0.0 '
                         'means transparent and 1.0 is completely visible. '
                         '[%(default)s]')

    bg.add_argument('--bg_offset', type=float, default=0.5,
                    help='The offset of the background image. [%(default)s]')

    bg.add_argument('--bg_interpolation',
                    default='nearest', choices={'linear', 'nearest'},
                    help='Interpolation mode for the background image. '
                         '[%(default)s]')

    bg.add_argument('--bg_color', nargs=3, type=float, default=(0, 0, 0),
                    help='The color of the overall background, behind '
                         'everything. Must be RGB values scaled between 0 and '
                         '1. [%(default)s]')

    # Peaks input file options
    peaks = p.add_argument_group('Peaks arguments')
    peaks.add_argument('--peaks', help='Peaks image file.')

    peaks.add_argument('--peaks_color', nargs=3, type=float,
                       help='Color used for peaks, as RGB values scaled '
                            'between 0 and 1. If None, then a RGB colormap is '
                            'used. [%(default)s]')

    peaks.add_argument('--peaks_width', default=1.0, type=float,
                       help='Width of peaks segments. [%(default)s]')

    peaks_scale = p.add_argument_group('Peaks scaling arguments', 'Choose '
                                       'between peaks values and arbitrary '
                                       'length.')
    peaks_scale_group = peaks_scale.add_mutually_exclusive_group()
    peaks_scale_group.add_argument('--peaks_values',
                                   help='Peaks values file.')

    peaks_scale_group.add_argument('--peaks_length', default=0.65, type=float,
                                   help='Length of the peaks segments. '
                                        '[%(default)s]')

    # fODF variance options
    var = p.add_argument_group('Variance arguments', 'For the visualization '
                               'of fodf uncertainty, the variance is used '
                               'as follow: mean + k * sqrt(variance), where '
                               'mean is the input fodf (in_fodf) and k is the '
                               'scaling factor (variance_k).')
    var.add_argument('--variance', help='FODF variance file.')
    var.add_argument('--variance_k', default=1, type=float,
                     help='Scaling factor (k) for the computation of the fodf '
                          'uncertainty. [%(default)s]')
    var.add_argument('--var_color', nargs=3, type=float, default=(1, 1, 1),
                     help='Color of variance outline. Must be RGB values '
                          'scaled between 0 and 1. [%(default)s]')

    return p


def _parse_args(parser):
    args = parser.parse_args()
    output = []
    if args.output:
        output.append(args.output)
    else:
        if args.silent:
            parser.error('Silent mode is enabled but no output is specified.'
                         'Specify an output with --output to use silent mode.')

    if args.peaks_values and not args.peaks:
        parser.error('Peaks values image supplied without peaks. Specify '
                     'a peaks image with --peaks to use this feature.')

    optional = [args.in_transparency_mask, args.mask, args.background,
                args.peaks, args.peaks_values]
    assert_inputs_exist(parser, args.in_fodf, optional)
    assert_outputs_exist(parser, args, output)
    assert_headers_compatible(parser, args.in_fodf, optional)

    return args


def _get_data_from_inputs(args):
    """
    Load data given by args. Perform checks to ensure dimensions agree
    between the data for mask, background, peaks and fODF.
    """

    fodf = nib.load(args.in_fodf).get_fdata(dtype=np.float32)
    data = {'fodf': fodf}
    if args.background:
        assert_same_resolution([args.background, args.in_fodf])
        bg = nib.load(args.background).get_fdata()
        data['bg'] = bg
    if args.in_transparency_mask:
        transparency_mask = get_data_as_mask(
            nib.load(args.in_transparency_mask), dtype=bool
        )
        data['transparency_mask'] = transparency_mask
    if args.mask:
        assert_same_resolution([args.mask, args.in_fodf])
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        data['mask'] = mask
    if args.peaks:
        assert_same_resolution([args.peaks, args.in_fodf])
        peaks = nib.load(args.peaks).get_fdata()
        if len(peaks.shape) == 4:
            last_dim = peaks.shape[-1]
            if last_dim % 3 == 0:
                npeaks = int(last_dim / 3)
                peaks = peaks.reshape((peaks.shape[:3] + (npeaks, 3)))
            else:
                raise ValueError('Peaks volume last dimension ({0}) cannot '
                                 'be reshaped as (npeaks, 3).'
                                 .format(peaks.shape[-1]))
        data['peaks'] = peaks
        if args.peaks_values:
            assert_same_resolution([args.peaks_values, args.in_fodf])
            peak_vals =\
                nib.load(args.peaks_values).get_fdata()
            data['peaks_values'] = peak_vals
    if args.variance:
        assert_same_resolution([args.variance, args.in_fodf])
        variance = nib.load(args.variance).get_fdata(dtype=np.float32)
        if len(variance.shape) == 3:
            variance = np.reshape(variance, variance.shape + (1,))
        if variance.shape != fodf.shape:
            raise ValueError('Dimensions mismatch between fODF {0} and '
                             'variance {1}.'
                             .format(fodf.shape, variance.shape))
        data['variance'] = variance

    return data


def main():
    parser = _build_arg_parser()
    args = _parse_args(parser)
    data = _get_data_from_inputs(args)
    sph = get_sphere(args.sphere)
    sh_order, full_basis = get_sh_order_and_fullness(data['fodf'].shape[-1])
    sh_basis, is_legacy = parse_sh_basis_arg(args)
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    actors = []

    # Retrieve the mask if supplied
    if 'mask' in data:
        mask = data['mask']
    else:
        mask = None

    if args.color_rgb:
        color_rgb = np.round(np.asarray(args.color_rgb) * 255)
    else:
        color_rgb = None

    variance = data['variance'] if args.variance else None
    var_color = np.asarray(args.var_color) * 255
    # Instantiate the ODF slicer actor
    odf_actor, var_actor = create_odf_slicer(data['fodf'], args.axis_name,
                                             args.slice_index, sph, sh_order,
                                             sh_basis, full_basis,
                                             args.scale, variance, mask,
                                             args.sph_subdivide,
                                             not args.radial_scale_off,
                                             not args.norm_off,
                                             args.colormap or color_rgb,
                                             variance_k=args.variance_k,
                                             variance_color=var_color,
                                             is_legacy=is_legacy)
    actors.append(odf_actor)

    # Instantiate a variance slicer actor if a variance image is supplied
    if 'variance' in data:
        actors.append(var_actor)

    # Instantiate a texture slicer actor if a background image is supplied
    if 'bg' in data:
        bg_actor = create_texture_slicer(data['bg'],
                                         args.axis_name,
                                         args.slice_index,
                                         mask,
                                         args.bg_range,
                                         args.bg_opacity,
                                         args.bg_offset,
                                         args.bg_interpolation)
        actors.append(bg_actor)

    # Instantiate a peaks slicer actor if peaks are supplied
    if 'peaks' in data:

        if 'peaks_values' in data:
            peaks_values = data['peaks_values']
        else:
            peaks_values =\
                np.ones(data['peaks'].shape[:-1]) * args.peaks_length
        peaks_actor = create_peaks_slicer(data['peaks'],
                                          args.axis_name,
                                          args.slice_index,
                                          peaks_values,
                                          mask,
                                          args.peaks_color,
                                          args.peaks_width,
                                          not full_basis)

        actors.append(peaks_actor)

    # Prepare and display the scene
    scene = create_scene(actors, args.axis_name,
                         args.slice_index,
                         data['fodf'].shape[:3],
                         args.win_dims[0] / args.win_dims[1],
                         bg_color=args.bg_color)

    mask_scene = None
    if 'transparency_mask' in data:
        mask_actor = create_texture_slicer(
            data['transparency_mask'].astype("uint8"),
            args.axis_name,
            args.slice_index,
            offset=0.0,
            )

        mask_scene = create_scene(
            [mask_actor],
            args.axis_name,
            args.slice_index,
            data['transparency_mask'].shape,
            args.win_dims[0] / args.win_dims[1],
            bg_color=args.bg_color)

    if not args.silent:
        create_interactive_window(
            scene, args.win_dims, args.interactor)

    if args.output:
        snapshots = snapshot_scenes(filter(None, [mask_scene, scene]),
                                    args.win_dims)
        _mask_arr = None
        if mask_scene:
            _mask_arr = any2grayscale(next(snapshots))

        image = compose_image(next(snapshots),
                              args.win_dims,
                              args.slice_index,
                              overlays_scene=_mask_arr)

        image.save(args.output)


if __name__ == '__main__':
    main()
