# -*- coding: utf-8 -*-

import numpy as np

from dipy.core.sphere import Sphere
from dipy.reconst.shm import sh_to_sf
from fury import window, actor


def create_odf_slicer(sh_fodf, mask, sphere, nb_subdivide,
                      sh_order, sh_basis, full_basis, axis_name,
                      scale, radial_scale, norm, colormap):
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
    fodf = sh_to_sf(sh_fodf, hi_res_sph, sh_order, dipy_basis_name)

    # Get mask if supplied, otherwise create a mask discarding empty voxels
    if mask is None:
        mask = np.linalg.norm(fodf, axis=-1) > 0.

    odf_actor = actor.odf_slicer(fodf[..., None, :], mask=mask[..., None],
                                 radial_scale=radial_scale, norm=norm,
                                 sphere=hi_res_rot_sph, colormap=colormap,
                                 scale=scale)

    odf_actor.display_extent(0, fodf.shape[0] - 1,
                             0, fodf.shape[1] - 1, 0, 0)

    return odf_actor


def create_texture_slicer(texture, value_range=None,
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


def create_scene(actors, grid_shape):
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


def render_scene(scene, window_size, interactor, output, silent):
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
