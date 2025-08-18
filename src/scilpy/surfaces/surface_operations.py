# -*- coding: utf-8 -*-

from dipy.io.stateful_surface import StatefulSurface
from nibabel.affines import apply_affine
import numpy as np

from scipy.ndimage import map_coordinates


def apply_transform(sfs, linear_transfo, target_reference,
                    deformation_img=None, inverse=False):
    """
    Apply transformation to a surface
        - Apply linear transformation with affine
        - Apply non linear transformation with ants_warp

    Parameters
    ----------
    sfs: StatefulSurface
        StatefulSurface containing the surface to be transformed
    
    linear_transfo: numpy.ndarray
        Transformation matrix to be applied
    deformation_img: nib.Nifti1Image
        Warp image from ANTs
    inverse: boolean
        Apply the inverse linear transformation.
    target_reference: nib.Nifti1Image
        Target reference image for the transformation.

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
    sfs.to_rasmm()
    sfs.vertices = apply_affine(affine, sfs.vertices)

    new_sfs = StatefulSurface(sfs.vertices, sfs.faces, target_reference,
                              space=sfs.space,
                              origin=sfs.origin)

    if deformation_img is not None:
        deformation_data = np.squeeze(deformation_img.get_fdata(dtype=np.float32))

        # Get vertices translation in voxel space, from the warp image
        new_sfs.to_vox()
        new_sfs.to_corner()

        tx = map_coordinates(deformation_data[..., 0], sfs.vertices.T, order=1)
        ty = map_coordinates(deformation_data[..., 1], sfs.vertices.T, order=1)
        tz = map_coordinates(deformation_data[..., 2], sfs.vertices.T, order=1)

    #     # Apply vertices translation in world coordinates
        new_sfs.to_rasmm()
        new_sfs.to_center()

        new_sfs.vertices[:, 0] += tx
        new_sfs.vertices[:, 1] += ty
        new_sfs.vertices[:, 2] += tz

    out_sfs = StatefulSurface(new_sfs.vertices, sfs.faces, target_reference,
                              space=sfs.space,
                              origin=sfs.origin)
    return out_sfs


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
