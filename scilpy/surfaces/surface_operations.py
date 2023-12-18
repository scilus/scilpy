# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import map_coordinates
import trimeshpy.vtk_util as vtk_u


def apply_transform(mesh, ants_affine, ants_warp=None):
    """
    Apply transformation to a surface
        - Apply linear transformation with affine
        - Apply non linear transformation with ants_warp

    Parameters
    ----------
    mesh: trimeshpy - Triangle Mesh VTK class
        Moving surface
    ants_affine: numpy.ndarray
        Transformation matrix to be applied
    ants_warp: nib.Nifti1Image
        Warp image from ANTs

    Returns
    -------
    mesh: trimeshpy - Triangle Mesh VTK class
        Surface moved
    """
    # Affine transformation
    inv_affine = np.linalg.inv(ants_affine)

    # Transform mesh vertices
    mesh.set_vertices(mesh.vertices_affine(inv_affine))

    # Flip triangle face, if needed
    if mesh.is_transformation_flip(inv_affine):
        mesh.set_triangles(mesh.triangles_face_flip())

    if ants_warp is not None:
        warp_img = np.squeeze(ants_warp.get_fdata(dtype=np.float32))

        # Get vertices translation in voxel space, from the warp image
        vts_vox = vtk_u.vtk_to_vox(mesh.get_vertices(), warp_img)
        tx = map_coordinates(warp_img[..., 0], vts_vox.T, order=1)
        ty = map_coordinates(warp_img[..., 1], vts_vox.T, order=1)
        tz = map_coordinates(warp_img[..., 2], vts_vox.T, order=1)

        # Apply vertices translation in world coordinates
        mesh.set_vertices(mesh.get_vertices() + np.array([tx, ty, tz]).T)

    return mesh


def flip(mesh, axes):
    """
    Apply flip to a surface

    Parameters
    ----------
    mesh: trimeshpy - Triangle Mesh VTK class
        Moving surface
    axes: list
        Axes (or normal orientation) you want to flip

    Returns
    -------
    mesh: trimeshpy - Triangle Mesh VTK class
        Surface flipped
    """
    # Flip axes
    flip = (-1 if 'x' in axes else 1,
            -1 if 'y' in axes else 1,
            -1 if 'z' in axes else 1)
    tris, vts = mesh.flip_triangle_and_vertices(flip)
    mesh.set_vertices(vts)
    mesh.set_triangles(tris)

    # Reverse surface orientation
    if 'n' in axes:
        tris = mesh.triangles_face_flip()
        mesh.set_triangles(tris)

    return mesh
