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
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sh_to_sf
from fury import window, actor

from scilpy.io.utils import (add_sh_basis_args, add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.io.image import get_data_as_mask


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

    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, output)

    return args


def _crop_along_axis(data, index, axis_name):
    """
    Extract a 2-dimensional slice from a 3-dimensional data volume
    """
    if axis_name == 'sagittal':
        if index is None:
            return data[data.shape[0]//2, :, :]
        return data[index, :, :]
    elif axis_name == 'coronal':
        if index is None:
            return data[:, data.shape[1]//2, :]
        return data[:, index, :]
    elif axis_name == 'axial':
        if index is None:
            return data[:, :, data.shape[2]//2]
        return data[:, :, index]


def _get_data_from_inputs(args):
    """
    Load data given by args. Perform checks to ensure dimensions agree
    between the data for mask, background and fODF.
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

    grid_shape = data['fodf'].shape[:2]
    return data, grid_shape


def _initialize_odf_slicer(data_dict, sphere, nb_subdivide, sh_order, sh_basis,
                           full_basis, axis_name, scale, radial_scale, norm,
                           colormap):
    """
    Create a ODF slicer actor displaying fODF in the XY plane. To account for
    the transformation of an initial 2D grid in the plane corresponding to
    sagittal or coronal orientation to the XY plane (axial orientation),
    the SF are rotated in the new coordinate system.
    """
    # Create rotated sphere for fodf in axis orientation
    if axis_name == 'sagittal':
        rot = np.array([[0., 0., 1.],
                        [1., 0., 0.],
                        [0., 1., 0.]])
        rot_sph = Sphere(xyz=np.dot(sphere.vertices, rot))
    elif axis_name == 'coronal':
        rot = np.array([[1., 0., 0.],
                        [0., 0., -1.],
                        [0., 1., 0.]])
        rot_sph = Sphere(xyz=np.dot(sphere.vertices, rot))
    else:
        rot_sph = sphere

    # Subdivide the spheres if nb_subdivide is provided
    if nb_subdivide is not None:
        hi_res_sph = sphere.subdivide(nb_subdivide)
        hi_res_rot_sph = rot_sph.subdivide(nb_subdivide)
    else:
        hi_res_sph = sphere
        hi_res_rot_sph = sphere

    # Convert SH coefficients to SF coefficients
    dipy_basis_name = sh_basis + '_full' if full_basis else sh_basis
    fodf = sh_to_sf(data_dict['fodf'], hi_res_sph, sh_order, dipy_basis_name)

    # Get mask if supplied, otherwise create a mask discarding empty voxels
    if 'mask' in data_dict:
        mask = data_dict['mask']
    else:
        mask = np.linalg.norm(fodf, axis=-1) > 0.

    odf_actor = actor.odf_slicer(fodf[..., None, :], mask=mask[..., None],
                                 radial_scale=radial_scale, norm=norm,
                                 sphere=hi_res_rot_sph, colormap=colormap,
                                 scale=scale)

    odf_actor.display_extent(0, fodf.shape[0] - 1,
                             0, fodf.shape[1] - 1, 0, 0)

    return odf_actor


def _initialize_texture_slicer(texture, value_range=None,
                               offset=None, interpolation=None):
    """
    Create a texture displayed behind the fODF. The texture is applied on a
    plane with a given offset for the fODF grid. The texture color values are
    linearly interpolated inside value_range.
    """
    # offset = None defaults to 0.5
    if offset is None:
        offset = 0.5

    # interpolation = None defaults to 'nearest'
    if interpolation is None:
        interpolation = 'nearest'

    affine = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, offset],
                       [0.0, 0.0, 0.0, 1.0]])

    slicer_actor =\
        actor.slicer(texture[..., None], affine=affine,
                     value_range=value_range, interpolation=interpolation)
    slicer_actor.display_extent(0, texture.shape[0] - 1,
                                0, texture.shape[1] - 1, 0, 0)

    return slicer_actor


def _initialize_scene_and_camera(actors, grid_shape):
    """
    Create a 3D scene containing actors fitting inside a 2-dimensional grid in
    the XY plane. The coordinate system of the scene is right-handed. The
    camera's up vector is in direction +Y and its view vector is in
    direction +Z. The projection is parallel.
    """
    # Configure camera
    if grid_shape[0] > grid_shape[1]:
        zoom_factor = 2.0 / grid_shape[0]
        eye_distance = grid_shape[0]
    else:
        zoom_factor = 2.0 / grid_shape[1]
        eye_distance = grid_shape[1]

    view_position = [(grid_shape[0] - 1) / 2.0,
                     (grid_shape[1] - 1) / 2.0,
                     -eye_distance]
    view_center = [(grid_shape[0] - 1) / 2.0,
                   (grid_shape[1] - 1) / 2.0,
                   0.0]
    view_up = [0.0, 1.0, 0.0]

    scene = window.Scene()
    scene.projection('parallel')
    scene.set_camera(position=view_position,
                     focal_point=view_center,
                     view_up=view_up)
    scene.zoom(zoom_factor)

    # Add actors to the scene
    for actor in actors:
        scene.add(actor)

    return scene


def _render_scene(scene, window_size, interactor, output, silent):
    """
    Render a scene. If a output is supplied, a snapshot of the rendered
    scene is taken.
    """
    if not silent:
        showm = window.ShowManager(scene, size=window_size,
                                   reset_camera=False,
                                   interactor_style=interactor)
        showm.initialize()
        showm.start()

    if output:
        out_img = window.snapshot(scene, size=window_size, fname=output)


def main():
    parser = _build_arg_parser()
    args = _parse_args(parser)
    data, grid_shape = _get_data_from_inputs(args)
    sph = get_sphere(args.sphere)

    actors = []
    # Instantiate the ODF slicer actor
    odf_actor = _initialize_odf_slicer(data, sph, args.sph_subdivide,
                                       args.sh_order, args.sh_basis,
                                       args.full_basis, args.axis_name,
                                       args.scale, not args.radial_scale_off,
                                       not args.norm_off, args.colormap)
    actors.append(odf_actor)

    # Instantiate a texture slicer actor if a background image is supplied
    if 'bg' in data:
        bg_actor = _initialize_texture_slicer(data['bg'],
                                              args.bg_range,
                                              args.bg_offset,
                                              args.bg_interpolation)
        actors.append(bg_actor)

    # Prepare and display the scene
    scene = _initialize_scene_and_camera(actors, grid_shape)
    _render_scene(scene, args.win_dims, args.interactor,
                  args.output, args.silent)


if __name__ == '__main__':
    main()
