#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 2-dimensional fODF slice loaded from disk.

Given a SH coefficients image, this script displays a slice in the
orientation specified by the user. The user can also add a background
on top of which the fODF are to be displayed.
"""

import argparse

import nibabel as nib
import numpy as np

from dipy.data import get_sphere

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

    p.add_argument('--slice_index', type=int,
                   help='Index of the slice to visualize along a given axis.')

    # Window configuration options
    p.add_argument('--win_dims', nargs=2, metavar=('WIDTH', 'HEIGHT'),
                   default=[768, 768], type=int,
                   help='The dimensions for the vtk window.')

    p.add_argument('--interactor', default='trackball',
                   choices={'image', 'trackball'},
                   help='Specify interactor mode for vtk window.')

    p.add_argument('--axis_name', default='axial', type=str,
                   choices={'axial', 'coronal', 'sagittal'},
                   help='Name of the axis to visualize.')

    p.add_argument('--silent', action='store_true',
                   help='Disable interactive visualization.')

    p.add_argument('--output', help='Path to output file.')

    add_overwrite_arg(p)

    # Optional FODF personalization arguments
    p.add_argument('--sh_order', type=int, default=8,
                   help='The SH order of the input fODF.')

    add_sh_basis_args(p)

    p.add_argument('--full_basis', action='store_true',
                   help='Use full SH basis to reconstruct fODF from '
                   'coefficients.')

    sphere_choices = {'symmetric362', 'symmetric642', 'symmetric724',
                      'repulsion724', 'repulsion100', 'repulsion200'}
    p.add_argument('--sphere', default='symmetric724', choices=sphere_choices,
                   help='Name of the sphere used to reconstruct SF.')

    p.add_argument('--sph_subdivide', type=int,
                   help='Number of subdivisions for given sphere.')

    p.add_argument('--mask',
                   help='Optional mask file. Only fODF inside '
                        'the mask are displayed.')

    p.add_argument('--colormap', default='jet',
                   help='Colormap for the ODF slicer.')

    p.add_argument('--scale', default=0.5, type=float,
                   help='Scaling factor for FODF.')

    p.add_argument('--radial_scale_off', action='store_true',
                   help='Disable radial scale for ODF slicer.')

    p.add_argument('--norm_off', action='store_true',
                   help='Disable normalization of ODF slicer.')

    # Background image options
    p.add_argument('--background',
                   help='Background image file.')

    p.add_argument('--bg_range', nargs=2, metavar=('MIN', 'MAX'), type=float,
                   help='The range of values mapped to range [0, 1] '
                        'for background image.')

    p.add_argument('--bg_offset', type=float,
                   help='The offset of the background image.')

    p.add_argument('--bg_interpolation', choices={'linear', 'nearest'},
                   help='Interpolation mode for the background image.')

    # Peaks input file options
    p.add_argument('--peaks',
                   help='Peaks image file.')

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
    else:
        if args.bg_range:
            parser.error('Background range is specified but no background '
                         'image is specified. Specify a background image '
                         'with --background to use this feature.')
        if args.bg_offset:
            parser.error('Background image offset is specified but no '
                         'background image is specified. Specify a background '
                         'image with --background to use this feature.')
        if args.bg_interpolation:
            parser.error('Background image interpolation is specified but no '
                         'background image is specified. Specify a background '
                         'image with --background to use this feature.')
    if args.peaks:
        if args.full_basis:
            # FURY doesn't support asymmetric peaks visualization
            parser.error('Cannot use peaks file with full basis: '
                         'Asymmetric peaks visualization is not available.')
        inputs.append(args.peaks)

    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, output)

    return args


def _crop_along_axis(data, index, axis_name):
    """
    Extract a 2-dimensional slice from a 3-dimensional data volume
    """
    if axis_name == 'sagittal':
        if index is None:
            data_slice = data[data.shape[0]//2, :, :]
        else:
            data_slice = data[index, :, :]
        return data_slice[None, ...]
    elif axis_name == 'coronal':
        if index is None:
            data_slice = data[:, data.shape[1]//2, :]
        else:
            data_slice = data[index, :, :]
        return data_slice[:, None, ...]
    elif axis_name == 'axial':
        if index is None:
            data_slice = data[:, :, data.shape[2]//2]
        else:
            data_slice = data[:, :, index]
        return data_slice[:, :, None]


def _get_data_from_inputs(args):
    """
    Load data given by args. Perform checks to ensure dimensions agree
    between the data for mask, background, peaks and fODF.
    """
    fodf = nib.nifti1.load(args.in_fodf).get_fdata(dtype=np.float32)
    data = {'fodf': _crop_along_axis(fodf, args.slice_index,
                                     args.axis_name)}
    if args.background:
        bg = nib.nifti1.load(args.background).get_fdata(dtype=np.float32)
        if bg.shape[:3] != fodf.shape[:-1]:
            raise ValueError('Background dimensions {0} do not agree with fODF'
                             ' dimensions {1}.'.format(bg.shape, fodf.shape))
        data['bg'] = _crop_along_axis(bg, args.slice_index,
                                      args.axis_name)
    if args.mask:
        mask = get_data_as_mask(nib.nifti1.load(args.mask), dtype=bool)
        if mask.shape != fodf.shape[:-1]:
            raise ValueError('Mask dimensions {0} do not agree with fODF '
                             'dimensions {1}.'.format(mask.shape, fodf.shape))
        data['mask'] = _crop_along_axis(mask, args.slice_index,
                                        args.axis_name)
    if args.peaks:
        peaks = nib.nifti1.load(args.peaks).get_fdata(dtype=np.float32)
        if peaks.shape[:3] != fodf.shape[:-1]:
            raise ValueError('Peaks volume dimensions {0} do not agree '
                             'with fODF dimensions {1}.'.format(bg.shape,
                                                                fodf.shape))
        data['peaks'] = _crop_along_axis(peaks, args.slice_index,
                                         args.axis_name)

    grid_shape = data['fodf'].shape[:3]
    return data, grid_shape


def main():
    parser = _build_arg_parser()
    args = _parse_args(parser)
    data, grid_shape = _get_data_from_inputs(args)
    sph = get_sphere(args.sphere)

    actors = []

    # Retrieve the mask if supplied
    if 'mask' in data:
        mask = data['mask']
    else:
        mask = None

    # Instantiate the ODF slicer actor
    odf_actor = create_odf_slicer(data['fodf'], mask, sph,
                                  args.sph_subdivide, args.sh_order,
                                  args.sh_basis, args.full_basis,
                                  args.axis_name, args.scale,
                                  not args.radial_scale_off,
                                  not args.norm_off, args.colormap)
    actors.append(odf_actor)

    # Instantiate a texture slicer actor if a background image is supplied
    if 'bg' in data:
        bg_actor = create_texture_slicer(data['bg'],
                                         args.bg_range,
                                         args.axis_name,
                                         args.bg_offset,
                                         args.bg_interpolation)
        actors.append(bg_actor)

    # Instantiate a peaks slicer actor if peaks are supplied
    if 'peaks' in data:
        peaks_actor = create_peaks_slicer(data['peaks'],
                                          args.axis_name,
                                          mask)
        actors.append(peaks_actor)

    # Prepare and display the scene
    scene = create_scene(actors, args.axis_name, grid_shape)
    render_scene(scene, args.win_dims, args.interactor,
                 args.output, args.silent)


if __name__ == '__main__':
    main()
