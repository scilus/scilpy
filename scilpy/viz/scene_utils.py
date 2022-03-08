# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np

from dipy.reconst.shm import sh_to_sf_matrix
from fury import window, actor
from fury.colormap import distinguishable_colormap
import vtk

from scilpy.io.utils import snapshot
from scilpy.reconst.bingham import bingham_to_sf


class CamParams(Enum):
    """
    Enum containing camera parameters
    """
    VIEW_POS = 'view_position'
    VIEW_CENTER = 'view_center'
    VIEW_UP = 'up_vector'
    ZOOM_FACTOR = 'zoom_factor'


def initialize_camera(orientation, slice_index, volume_shape):
    """
    Initialize a camera for a given orientation.
    """
    camera = {}
    # Tighten the view around the data
    camera[CamParams.ZOOM_FACTOR] = 2.0 / max(volume_shape)
    # heuristic for setting the camera position at a distance
    # proportional to the scale of the scene
    eye_distance = max(volume_shape)
    if orientation == 'sagittal':
        if slice_index is None:
            slice_index = volume_shape[0] // 2
        camera[CamParams.VIEW_POS] = np.array([-eye_distance,
                                               (volume_shape[1] - 1) / 2.0,
                                               (volume_shape[2] - 1) / 2.0])
        camera[CamParams.VIEW_CENTER] = np.array([slice_index,
                                                  (volume_shape[1] - 1) / 2.0,
                                                  (volume_shape[2] - 1) / 2.0])
        camera[CamParams.VIEW_UP] = np.array([0.0, 0.0, 1.0])
    elif orientation == 'coronal':
        if slice_index is None:
            slice_index = volume_shape[1] // 2
        camera[CamParams.VIEW_POS] = np.array([(volume_shape[0] - 1) / 2.0,
                                               eye_distance,
                                               (volume_shape[2] - 1) / 2.0])
        camera[CamParams.VIEW_CENTER] = np.array([(volume_shape[0] - 1) / 2.0,
                                                  slice_index,
                                                  (volume_shape[2] - 1) / 2.0])
        camera[CamParams.VIEW_UP] = np.array([0.0, 0.0, 1.0])
    elif orientation == 'axial':
        if slice_index is None:
            slice_index = volume_shape[2] // 2
        camera[CamParams.VIEW_POS] = np.array([(volume_shape[0] - 1) / 2.0,
                                               (volume_shape[1] - 1) / 2.0,
                                               -eye_distance])
        camera[CamParams.VIEW_CENTER] = np.array([(volume_shape[0] - 1) / 2.0,
                                                  (volume_shape[1] - 1) / 2.0,
                                                  slice_index])
        camera[CamParams.VIEW_UP] = np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError('Invalid axis name: {0}'.format(orientation))
    return camera


def set_display_extent(slicer_actor, orientation, volume_shape, slice_index):
    """
    Set the display extent for a fury actor in ``orientation``.
    """
    if orientation == 'sagittal':
        if slice_index is None:
            slice_index = volume_shape[0] // 2
        slicer_actor.display_extent(slice_index, slice_index,
                                    0, volume_shape[1],
                                    0, volume_shape[2])
    elif orientation == 'coronal':
        if slice_index is None:
            slice_index = volume_shape[1] // 2
        slicer_actor.display_extent(0, volume_shape[0],
                                    slice_index, slice_index,
                                    0, volume_shape[2])
    elif orientation == 'axial':
        if slice_index is None:
            slice_index = volume_shape[2] // 2
        slicer_actor.display_extent(0, volume_shape[0],
                                    0, volume_shape[1],
                                    slice_index, slice_index)
    else:
        raise ValueError('Invalid axis name : {0}'.format(orientation))


def create_odf_slicer(sh_fodf, mask, sphere, nb_subdivide,
                      sh_order, sh_basis, full_basis, orientation,
                      scale, radial_scale, norm, colormap, slice_index):
    """
    Create a ODF slicer actor displaying a fODF slice. The input volume is a
    3-dimensional grid containing the SH coefficients of the fODF for each
    voxel at each voxel, with the grid dimension having a size of 1 along the
    axis corresponding to the selected orientation.
    """
    # Subdivide the spheres if nb_subdivide is provided
    if nb_subdivide is not None:
        sphere = sphere.subdivide(nb_subdivide)

    # SH coefficients to SF coefficients matrix
    B_mat = sh_to_sf_matrix(sphere, sh_order, sh_basis,
                            full_basis, return_inv=False)

    odf_actor = actor.odf_slicer(sh_fodf, mask=mask, norm=norm,
                                 radial_scale=radial_scale,
                                 sphere=sphere,
                                 colormap=colormap,
                                 scale=scale, B_matrix=B_mat)
    set_display_extent(odf_actor, orientation, sh_fodf.shape[:3], slice_index)

    return odf_actor


def _get_affine_for_texture(orientation, offset):
    """
    Get the affine transformation to apply to the texture
    to offset it from the fODF grid.
    """
    if orientation == 'sagittal':
        v = np.array([offset, 0.0, 0.0])
    elif orientation == 'coronal':
        v = np.array([0.0, -offset, 0.0])
    elif orientation == 'axial':
        v = np.array([0.0, 0.0, offset])
    else:
        raise ValueError('Invalid axis name : {0}'.format(orientation))

    affine = np.identity(4)
    affine[0:3, 3] = v
    return affine


def create_texture_slicer(texture, mask, slice_index, value_range=None,
                          orientation='axial', opacity=1.0, offset=0.5,
                          interpolation='nearest'):
    """
    Create a texture displayed behind the fODF. The texture is applied on a
    plane with a given offset for the fODF grid.
    """
    affine = _get_affine_for_texture(orientation, offset)

    if mask is not None:
        masked_texture = np.zeros_like(texture)
        masked_texture[mask] = texture[mask]
    else:
        masked_texture = texture

    if value_range:
        masked_texture = (masked_texture - value_range[0])
                         / value_range[1] * 255
        masked_texture = np.clip(masked_texture, 0 , 255)

    slicer_actor = actor.slicer(masked_texture, affine=affine,
                                value_range=value_range,
                                opacity=opacity,
                                interpolation=interpolation)
    set_display_extent(slicer_actor, orientation, texture.shape, slice_index)
    return slicer_actor


def create_peaks_slicer(data, orientation, slice_index, peak_values=None,
                        mask=None, color=None, peaks_width=1.0,
                        symmetric=False):
    """
    Create a peaks slicer actor rendering a slice of the fODF peaks
    """
    # Normalize input data
    norm = np.linalg.norm(data, axis=-1)
    data[norm > 0] /= norm[norm > 0].reshape((-1, 1))

    # Instantiate peaks slicer
    peaks_slicer = actor.peak_slicer(data, peaks_values=peak_values,
                                     mask=mask, colors=color,
                                     linewidth=peaks_width,
                                     symmetric=symmetric)
    set_display_extent(peaks_slicer, orientation, data.shape, slice_index)

    return peaks_slicer


def create_bingham_slicer(data, orientation, slice_index,
                          sphere, color_per_lobe=False):
    """
    Create a bingham fit slicer using a combination of odf_slicer actors

    Parameters
    ----------
    data: ndarray (X, Y, Z, 9 * nb_lobes)
        The Bingham volume.
    orientation: string
        One of 'sagittal', 'coronal', 'axial'.
    slice_index: int
        Index of the slice of interest along the chosen orientation.
    sphere: DIPY Sphere
        Sphere used for visualization.
    color_per_lobe: bool, optional
        If true, each Bingham distribution is colored using a disting color.
        Else, Bingham distributions are colored by their orientation.

    Return
    ------
    actors: list of fury odf_slicer actors
        ODF slicer actors representing the Bingham distributions.
    """
    shape = data.shape
    nb_lobes = shape[-2]
    colors = [c * 255 for i, c in zip(range(nb_lobes),
                                      distinguishable_colormap())]

    # lmax norm for normalization
    lmaxnorm = np.max(np.abs(data[..., 0]), axis=-1)
    bingham_sf = bingham_to_sf(data, sphere.vertices)

    actors = []
    for nn in range(nb_lobes):
        sf = bingham_sf[..., nn, :]
        sf[lmaxnorm > 0] /= lmaxnorm[lmaxnorm > 0][:, None]
        color = colors[nn] if color_per_lobe else None
        odf_actor = actor.odf_slicer(sf, sphere=sphere, norm=False,
                                     colormap=color)
        set_display_extent(odf_actor, orientation, shape[:3], slice_index)
        actors.append(odf_actor)

    return actors


def create_tube_with_radii(positions, radii, error, error_coloring=False,
                           wireframe=False):
    # Generate the polydata from the centroids
    joint_count = len(positions)
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(joint_count)
    for j in range(joint_count):
        pts.InsertPoint(j, positions[j])
        lines.InsertCellPoint(j)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pts)
    polydata.SetLines(lines)

    # Generate the coloring from either the labels or the fitting error
    colors_arr = vtk.vtkFloatArray()
    for i in range(joint_count):
        if error_coloring:
            colors_arr.InsertNextValue(error[i])
        else:
            colors_arr.InsertNextValue(len(error) - 1 - i)
    colors_arr.SetName("colors")
    polydata.GetPointData().AddArray(colors_arr)

    # Generate the radii array for VTK
    radii_arr = vtk.vtkFloatArray()
    for i in range(joint_count):
        radii_arr.InsertNextValue(radii[i])
    radii_arr.SetName("radii")
    polydata.GetPointData().SetScalars(radii_arr)

    # Tube filter for the rendering with varying radii
    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputData(polydata)
    tubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tubeFilter.SetNumberOfSides(25)
    tubeFilter.CappingOn()

    # Map the coloring to the tube filter
    tubeFilter.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tubeFilter.GetOutputPort())
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("colors")
    if error_coloring:
        mapper.SetScalarRange(0, max(error))
    else:
        mapper.SetScalarRange(0, len(error))

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()

    return actor


def create_scene(actors, orientation, slice_index, volume_shape):
    """
    Create a 3D scene containing actors fitting inside a grid. The camera is
    placed based on the orientation supplied by the user. The projection mode
    is parallel.
    """
    # Configure camera
    camera = initialize_camera(orientation, slice_index, volume_shape)

    scene = window.Scene()
    scene.projection('parallel')
    scene.set_camera(position=camera[CamParams.VIEW_POS],
                     focal_point=camera[CamParams.VIEW_CENTER],
                     view_up=camera[CamParams.VIEW_UP])
    scene.zoom(camera[CamParams.ZOOM_FACTOR])

    # Add actors to the scene
    for curr_actor in actors:
        scene.add(curr_actor)

    return scene


def render_scene(scene, window_size, interactor,
                 output, silent, title='Viewer'):
    """
    Render a scene. If a output is supplied, a snapshot of the rendered
    scene is taken.
    """
    if not silent:
        showm = window.ShowManager(scene, title=title,
                                   size=window_size,
                                   reset_camera=False,
                                   interactor_style=interactor)

        showm.initialize()
        showm.start()

    if output:
        snapshot(scene, output, size=window_size)
