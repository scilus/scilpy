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
import logging
import warnings

import nibabel as nib
import numpy as np

from dipy.data import get_sphere
from dipy.reconst.shm import order_from_ncoef

from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.io.utils import (add_sh_basis_args, add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.io.image import get_data_as_mask
from scilpy.viz.scene_utils import (create_odf_slicer, create_texture_slicer,
                                    create_peaks_slicer, create_scene,
                                    render_scene)


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
                   choices={'axial', 'coronal', 'sagittal'},
                   help='Name of the axis to visualize. [%(default)s]')

    p.add_argument('--silent', action='store_true',
                   help='Disable interactive visualization.')

    p.add_argument('--output', help='Path to output file.')

    add_overwrite_arg(p)

    # Optional FODF personalization arguments
    add_sh_basis_args(p)

    sphere_choices = {'symmetric362', 'symmetric642', 'symmetric724',
                      'repulsion724', 'repulsion100', 'repulsion200'}
    p.add_argument('--sphere', default='symmetric724', choices=sphere_choices,
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
                   help='Scaling factor for FODF. [%(default)s]')

    p.add_argument('--radial_scale_off', action='store_true',
                   help='Disable radial scale for ODF slicer.')

    p.add_argument('--norm_off', action='store_true',
                   help='Disable normalization of ODF slicer.')

    # Background image options
    p.add_argument('--background',
                   help='Background image file. If RGB, values must '
                        'be between 0 and 255.')

    p.add_argument('--bg_range', nargs=2, metavar=('MIN', 'MAX'), type=float,
                   help='The range of values mapped to range [0, 1] '
                        'for background image. [(bg.min(), bg.max())]')

    p.add_argument('--bg_opacity', type=float, default=1.0,
                   help='The opacity of the background image. Opacity of 0.0 '
                        'means transparent and 1.0 is completely visible. '
                        '[%(default)s]')

    p.add_argument('--bg_offset', type=float, default=0.5,
                   help='The offset of the background image. [%(default)s]')

    p.add_argument('--bg_interpolation',
                   default='nearest', choices={'linear', 'nearest'},
                   help='Interpolation mode for the background image. '
                        '[%(default)s]')

    # Peaks input file options
    p.add_argument('--peaks',
                   help='Peaks image file.')

    p.add_argument('--peaks_color', nargs=3, type=float,
                   help='Color used for peaks. If None, '
                        'then a RGB colormap is used. [%(default)s]')

    p.add_argument('--peaks_width', default=1.0, type=float,
                   help='Width of peaks segments. [%(default)s]')

    peaks_scale_group = p.add_mutually_exclusive_group()
    peaks_scale_group.add_argument('--peaks_values',
                                   help='Peaks values file.')

    peaks_scale_group.add_argument('--peaks_length', default=0.65, type=float,
                                   help='Length of the peaks segments. '
                                        '[%(default)s]')

    return p


def _parse_args(parser):
    args = parser.parse_args()
    inputs = []
    output = []
    inputs.append(args.in_fodf)
    if args.output:
        output.append(args.output)
    else:
        if args.silent:
            parser.error('Silent mode is enabled but no output is specified.'
                         'Specify an output with --output to use silent mode.')
    if args.mask:
        inputs.append(args.mask)
    if args.background:
        inputs.append(args.background)

    if args.peaks:
        inputs.append(args.peaks)
        if args.peaks_values:
            inputs.append(args.peaks_values)
    else:
        if args.peaks_values:
            parser.error('Peaks values image supplied without peaks. Specify '
                         'a peaks image with --peaks to use this feature.')

    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, output)

    return args


def _get_data_from_inputs(args):
    """
    Load data given by args. Perform checks to ensure dimensions agree
    between the data for mask, background, peaks and fODF.
    """
    fodf = nib.nifti1.load(args.in_fodf).get_fdata(dtype=np.float32)
    data = {'fodf': fodf}
    if args.background:
        bg = nib.nifti1.load(args.background).get_fdata(dtype=np.float32)
        if bg.shape[:3] != fodf.shape[:-1]:
            raise ValueError('Background dimensions {0} do not agree with fODF'
                             ' dimensions {1}.'.format(bg.shape, fodf.shape))
        data['bg'] = bg
    if args.mask:
        mask = get_data_as_mask(nib.nifti1.load(args.mask), dtype=bool)
        if mask.shape != fodf.shape[:-1]:
            raise ValueError('Mask dimensions {0} do not agree with fODF '
                             'dimensions {1}.'.format(mask.shape, fodf.shape))
        data['mask'] = mask
    if args.peaks:
        peaks = nib.nifti1.load(args.peaks).get_fdata(dtype=np.float32)
        if peaks.shape[:3] != fodf.shape[:-1]:
            raise ValueError('Peaks volume dimensions {0} do not agree '
                             'with fODF dimensions {1}.'.format(bg.shape,
                                                                fodf.shape))
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
            peak_vals =\
                nib.nifti1.load(args.peaks_values).get_fdata(dtype=np.float32)
            if peak_vals.shape[:3] != fodf.shape[:-1]:
                raise ValueError('Peaks volume dimensions {0} do not agree '
                                 'with fODF dimensions {1}.'
                                 .format(peak_vals.shape, fodf.shape))
            data['peaks_values'] = peak_vals

    return data


def main():
    parser = _build_arg_parser()
    args = _parse_args(parser)
    data = _get_data_from_inputs(args)
    sph = get_sphere(args.sphere)
    sh_order, full_basis = get_sh_order_and_fullness(data['fodf'].shape[-1])

    actors = []

    # Retrieve the mask if supplied
    if 'mask' in data:
        mask = data['mask']
    else:
        mask = None

    # Instantiate the ODF slicer actor
    odf_actor = create_odf_slicer(data['fodf'], args.axis_name,
                                  args.slice_index, mask, sph,
                                  args.sph_subdivide, sh_order,
                                  args.sh_basis, full_basis,
                                  args.scale,
                                  not args.radial_scale_off,
                                  not args.norm_off, args.colormap)
    actors.append(odf_actor)

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
        peaks_values = None
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
                         data['fodf'].shape[:3])
    render_scene(scene, args.win_dims, args.interactor,
                 args.output, args.silent)


if __name__ == '__main__':
    main()
