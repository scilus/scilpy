# -*- coding: utf-8 -*-

from nibabel.affines import apply_affine
import numpy as np

from scipy.ndimage import map_coordinates


def apply_transform(mesh, linear_transfo, deformation_img=None, inverse=False):
    """
    Apply transformation to a surface
        - Apply linear transformation with affine
        - Apply non linear transformation with ants_warp

    Parameters
    ----------
    mesh: trimeshpy - Triangle Mesh VTK class
        Moving surface
    linear_transfo: numpy.ndarray
        Transformation matrix to be applied
    deformation_img: nib.Nifti1Image
        Warp image from ANTs
    inverse: boolean
        Apply the inverse linear transformation.

    Returns
    -------
    mesh: trimeshpy - Triangle Mesh VTK class
        Surface moved
    """
    # Affine transformation
    affine = linear_transfo
    if inverse:
        affine = np.linalg.inv(linear_transfo)

    # Apply affine transformation
    flip = np.diag([-1, -1, 1, 1])
    mesh.set_vertices(apply_affine(flip, mesh.get_vertices()))
    mesh.set_vertices(apply_affine(affine, mesh.get_vertices()))

    # Flip triangle face, if needed
    if mesh.is_transformation_flip(affine):
        mesh.set_triangles(mesh.triangles_face_flip())

    if deformation_img is not None:
        deformation_data = np.squeeze(deformation_img.get_fdata(dtype=np.float32))

        # Get vertices translation in voxel space, from the warp image
        inv_affine = np.linalg.inv(deformation_img.affine)
        vts_vox = apply_affine(inv_affine, mesh.get_vertices())
        tx = map_coordinates(deformation_data[..., 0], vts_vox.T, order=1)
        ty = map_coordinates(deformation_data[..., 1], vts_vox.T, order=1)
        tz = map_coordinates(deformation_data[..., 2], vts_vox.T, order=1)

        # Apply vertices translation in world coordinates
        # LPS versus RAS (-tx, -ty)
        mesh.set_vertices(mesh.get_vertices() + np.array([-tx, -ty, tz]).T)

    mesh.set_vertices(apply_affine(flip, mesh.get_vertices()))
    mesh.update_polydata()

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
