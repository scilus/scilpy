# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np

import vtk
from dipy.reconst.shm import sh_to_sf_matrix, sh_to_sf
from fury import window, actor
from fury.colormap import distinguishable_colormap
from fury.utils import get_actor_from_polydata, numpy_to_vtk_image_data
from PIL import Image, ImageFont, ImageDraw

from scilpy.io.utils import snapshot
from scilpy.reconst.bingham import bingham_to_sf
from scilpy.viz.utils import get_colormap


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

    Parameters
    ----------
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    volume_shape : tuple
        Shape of the sliced volume.

    Returns
    -------
    camera : dict
        Dictionnary containing camera information.
    """
    camera = {}
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
        # Tighten the view around the data
        camera[CamParams.ZOOM_FACTOR] = 2.0 / max(volume_shape[1:])
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
        # Tighten the view around the data
        camera[CamParams.ZOOM_FACTOR] = 2.0 / max(
            [volume_shape[0], volume_shape[2]])
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
        # Tighten the view around the data
        camera[CamParams.ZOOM_FACTOR] = 2.0 / max(volume_shape[:2])
    else:
        raise ValueError('Invalid axis name: {0}'.format(orientation))
    return camera


def set_display_extent(slicer_actor, orientation, volume_shape, slice_index):
    """
    Set the display extent for a fury actor in ``orientation``.

    Parameters
    ----------
    slicer_actor : actor
        Slicer actor from Fury
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    volume_shape : tuple
        Shape of the sliced volume.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
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


def create_odf_slicer(sh_fodf, orientation, slice_index, mask, sphere,
                      nb_subdivide, sh_order, sh_basis, full_basis,
                      scale, radial_scale, norm, colormap, sh_variance=None,
                      variance_k=1, variance_color=(255, 255, 255),
                      is_legacy=True):
    """
    Create a ODF slicer actor displaying a fODF slice. The input volume is a
    3-dimensional grid containing the SH coefficients of the fODF for each
    voxel at each voxel, with the grid dimension having a size of 1 along the
    axis corresponding to the selected orientation.

    Parameters
    ----------
    sh_fodf : np.ndarray
        Spherical harmonics of fODF data.
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    mask : np.ndarray, optional
        Only the data inside the mask will be displayed. Defaults to None.
    sphere: DIPY Sphere
        Sphere used for visualization.
    nb_subdivide : int
        Number of subdivisions for given sphere. If None, uses the given sphere
        as is.
    sh_order : int
        Maximum spherical harmonics order.
    sh_basis : str
        Type of basis for the spherical harmonics.
    full_basis : bool
        Boolean indicating if the basis is full or not.
    scale : float
        Scaling factor for FODF.
    radial_scale : bool
        If True, enables radial scale for ODF slicer.
    norm : bool
        If True, enables normalization of ODF slicer.
    colormap : str
        Colormap for the ODF slicer. If None, a RGB colormap is used.
    sh_variance : np.ndarray, optional
        Spherical harmonics of the variance fODF data.
    variance_k : float, optional
        Factor that multiplies sqrt(variance).
    variance_color : tuple, optional
        Color of the variance fODF data, in RGB.
    is_legacy : bool, optional
        Whether or not the SH basis is in its legacy form.

    Returns
    -------
    odf_actor : actor.odf_slicer
        Fury object containing the odf information.
    """
    # Subdivide the spheres if nb_subdivide is provided
    if nb_subdivide is not None:
        sphere = sphere.subdivide(nb_subdivide)

    # SH coefficients to SF coefficients matrix
    B_mat = sh_to_sf_matrix(sphere, sh_order, sh_basis,
                            full_basis, return_inv=False, legacy=is_legacy)

    var_actor = None

    if sh_variance is not None:
        fodf = sh_to_sf(sh_fodf, sphere, sh_order, sh_basis,
                        full_basis=full_basis, legacy=is_legacy)
        fodf_var = sh_to_sf(sh_variance, sphere, sh_order, sh_basis,
                            full_basis=full_basis, legacy=is_legacy)
        fodf_uncertainty = fodf + variance_k * np.sqrt(np.clip(fodf_var, 0,
                                                               None))
        # normalise fodf and variance
        if norm:
            maximums = np.abs(np.append(fodf, fodf_uncertainty, axis=-1))\
                .max(axis=-1)
            fodf[maximums > 0] /= maximums[maximums > 0][..., None]
            fodf_uncertainty[maximums > 0] /= maximums[maximums > 0][..., None]

        odf_actor = actor.odf_slicer(fodf, mask=mask, norm=False,
                                     radial_scale=radial_scale,
                                     sphere=sphere, scale=scale,
                                     colormap=colormap)

        var_actor = actor.odf_slicer(fodf_uncertainty, mask=mask, norm=False,
                                     radial_scale=radial_scale,
                                     sphere=sphere, scale=scale,
                                     colormap=variance_color)
        var_actor.GetProperty().SetDiffuse(0.0)
        var_actor.GetProperty().SetAmbient(1.0)
        var_actor.GetProperty().SetFrontfaceCulling(True)
    else:
        odf_actor = actor.odf_slicer(sh_fodf, mask=mask, norm=norm,
                                     radial_scale=radial_scale,
                                     sphere=sphere,
                                     colormap=colormap,
                                     scale=scale, B_matrix=B_mat)
    set_display_extent(odf_actor, orientation, sh_fodf.shape[:3], slice_index)
    if var_actor is not None:
        set_display_extent(var_actor, orientation,
                           fodf_uncertainty.shape[:3], slice_index)

    return odf_actor, var_actor


def _get_affine_for_texture(orientation, offset):
    """
    Get the affine transformation to apply to the texture
    to offset it from the fODF grid.

    Parameters
    ----------
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    offset : float
        The offset of the texture image.

    Returns
    -------
    affine : np.ndarray
        The affine transformation.
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


def create_texture_slicer(texture, orientation, slice_index, mask=None,
                          value_range=None, opacity=1.0, offset=0.5,
                          interpolation='nearest'):
    """
    Create a texture displayed behind the fODF. The texture is applied on a
    plane with a given offset for the fODF grid.

    Parameters
    ----------
    texture : np.ndarray (3d or 4d)
        Texture image. Can be 3d for scalar data of 4d for RGB data, in which
        case the values must be between 0 and 255.
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    mask : np.ndarray, optional
        Only the data inside the mask will be displayed. Defaults to None.
    value_range : tuple (2,), optional
        The range of values mapped to range [0, 1] for the texture image. If
        None, it equals to (bg.min(), bg.max()). Defaults to None.
    opacity : float, optional
        The opacity of the texture image. Opacity of 0.0 means transparent and
        1.0 is completely visible. Defaults to 1.0.
    offset : float, optional
        The offset of the texture image. Defaults to 0.5.
    interpolation : str, optional
        Interpolation mode for the texture image. Choices are nearest or
        linear. Defaults to nearest.

    Returns
    -------
    slicer_actor : actor.slicer
        Fury object containing the texture information.
    """
    affine = _get_affine_for_texture(orientation, offset)

    if mask is not None:
        texture[np.where(mask == 0)] = 0

    if value_range:
        texture = np.clip((texture - value_range[0]) / value_range[1] * 255,
                          0, 255)

    slicer_actor = actor.slicer(texture, affine=affine,
                                opacity=opacity, interpolation=interpolation)
    set_display_extent(slicer_actor, orientation, texture.shape, slice_index)
    return slicer_actor


def create_peaks_slicer(data, orientation, slice_index, peak_values=None,
                        mask=None, color=None, peaks_width=1.0,
                        symmetric=False):
    """
    Create a peaks slicer actor rendering a slice of the fODF peaks

    Parameters
    ----------
    data : np.ndarray
        Peaks data.
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    peak_values : np.ndarray, optional
        Peaks values. Defaults to None.
    mask : np.ndarray, optional
        Only the data inside the mask will be displayed. Defaults to None.
    color : tuple (3,), optional
        Color used for peaks. If None, a RGB colormap is used. Defaults to
        None.
    peaks_width : float, optional
        Width of peaks segments. Defaults to 1.0.
    symmetric : bool, optional
        If True, peaks are drawn for both peaks_dirs and -peaks_dirs. Else,
        peaks are only drawn for directions given by peaks_dirs. Defaults to
        False.

    Returns
    -------
    slicer_actor : actor.peak_slicer
        Fury object containing the peaks information.
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
    orientation: str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
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


def contour_actor_from_image(
    img, axis, slice_index,
    contour_value=1.,
    color=[255, 0, 0],
    opacity=1.,
    linewidth=3.,
    smoothing_radius=0.
):
    """
    Get an isocontour actor from an image slice, at a defined value.

    Parameters
    ----------
    img : Nifti1Image
        Nifti volume (mask, binary image, labels).
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    axis : int
        Slicing axis
    contour_values : float
        Values at which to extract isocontours.
    color : tuple, list of int
        Color of the contour in RGB [0, 255].
    opacity: float
        Opacity of the contour.
    linewidth : float
        Thickness of the contour line.
    smoothing_radius : float
        Pre-smoothing to apply to the image before 
        computing the contour (in pixels).

    Returns
    -------
    actor : vtkActor
        Actor for the contour polydata.
    """

    mask_data = numpy_to_vtk_image_data(
        np.rot90(img.get_fdata().take([slice_index], axis).squeeze()))
    mask_data.SetOrigin(0, 0, 0)

    if smoothing_radius > 0:
        smoother = vtk.vtkImageGaussianSmooth()
        smoother.SetRadiusFactor(smoothing_radius)
        smoother.SetDimensionality(2)
        smoother.SetInputData(mask_data)
        smoother.Update()
        mask_data = smoother.GetOutput()

    marching_squares = vtk.vtkMarchingSquares()
    marching_squares.SetInputData(mask_data)
    marching_squares.SetValue(0, contour_value)
    marching_squares.Update()

    actor = get_actor_from_polydata(marching_squares.GetOutput())
    actor.GetMapper().ScalarVisibilityOff()
    actor.GetProperty().SetLineWidth(linewidth)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)

    position =[0, 0, 0]
    position[axis] = slice_index

    if axis == 0:
        actor.SetOrientation(90, 0, 90)
    elif axis == 1:
        actor.SetOrientation(90, 0, 0)

    actor.SetPosition(*position)

    return actor


def create_scene(actors, orientation, slice_index,
                 volume_shape, bg_color=(0, 0, 0)):
    """
    Create a 3D scene containing actors fitting inside a grid. The camera is
    placed based on the orientation supplied by the user. The projection mode
    is parallel.

    Parameters
    ----------
    actors : tab
        Ensemble of actors from Fury
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    volume_shape : tuple
        Shape of the sliced volume.
    bg_color: tuple, optional
        Background color expressed as RGB triplet in the range [0, 1].

    Returns
    -------
    scene : window.Scene()
        Object from Fury containing the 3D scene.
    """
    # Configure camera
    camera = initialize_camera(orientation, slice_index, volume_shape)

    scene = window.Scene()
    scene.background(bg_color)
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
                 output, silent, mask_scene=None, title='Viewer'):
    """
    Render a scene. If a output is supplied, a snapshot of the rendered
    scene is taken. If a mask is supplied, all values outside the mask are set
    to full transparency in the saved scene.

    Parameters
    ----------
    scene : window.Scene()
        3D scene to render.
    window_size : tuple (width, height)
        The dimensions for the vtk window.
    interactor : str
        Specify interactor mode for vtk window. Choices are image or trackball.
    output : str
        Path to output file.
    silent : bool
        If True, disable interactive visualization.
    mask_scene : window.Scene(), optional
        Transparency mask scene.
    title : str, optional
        Title of the scene. Defaults to Viewer.
    """
    if not silent:
        showm = window.ShowManager(scene, title=title,
                                   size=window_size,
                                   reset_camera=False,
                                   interactor_style=interactor)

        showm.initialize()
        showm.start()

    if output:
        if mask_scene is not None:
            # Create the screenshots
            scene_arr = window.snapshot(scene, size=window_size)
            mask_scene_arr = window.snapshot(mask_scene, size=window_size)
            # Create the target image
            out_img = create_canvas(*window_size, 0, 0, 1, 1)
            # Convert the mask scene data to grayscale and adjust for handling
            # with Pillow
            _mask_arr = rgb2gray4pil(mask_scene_arr)
            # Create the masked image
            draw_scene_at_pos(
                out_img, scene_arr, window_size, 0, 0, mask=_mask_arr
            )

            out_img.save(output)
        else:
            snapshot(scene, output, size=window_size)


def screenshot_slice(img, axis_name, slice_ids, size):
    """Take a screenshot of the given image with the appropriate slice data at
    the provided slice indices.

    Parameters
    ----------
    img : nib.Nifti1Image
        Volume image.
    axis_name : str
        Slicing axis name.
    slice_ids : array-like
        Slice indices.
    size : array-like
        Size of the screenshot image (pixels).

    Returns
    -------
    scene_container : list
        Scene screenshot data container.
    """

    scene_container = []

    for idx in slice_ids:

        slice_actor = create_texture_slicer(
            img.get_fdata(), axis_name, idx, offset=0.0
        )
        scene = create_scene([slice_actor], axis_name, idx, img.shape)
        scene_arr = window.snapshot(scene, size=size)
        scene_container.append(scene_arr)

    return scene_container


def screenshot_contour(bin_img, axis_name, slice_ids, size):
    """Take a screenshot of the given binary image countour with the 
    appropriate slice data at the provided slice indices.

    Parameters
    ----------
    bin_img : nib.Nifti1Image
        Binary volume image.
    axis_name : str
        Slicing axis name.
    slice_ids : array-like
        Slice indices.
    size : array-like
        Size of the screenshot image (pixels).

    Returns
    -------
    scene_container : list
        Scene screenshot data container.
    """
    scene_container = []

    if axis_name == "axial":
        ax_idx = 2
    elif axis_name == "coronal":
        ax_idx = 1
    elif axis_name == "sagittal":
        ax_idx = 0

    image_size_2d = list(bin_img.shape)
    image_size_2d[ax_idx] = 1

    for idx in slice_ids:
        actor = contour_actor_from_image(
            bin_img, ax_idx, idx, color=[255, 255, 255])

        scene = create_scene([actor], axis_name, idx, image_size_2d)
        scene_arr = window.snapshot(scene, size=size)
        scene_container.append(scene_arr)

    return scene_container


def check_mosaic_layout(img_count, rows, cols):
    """Check whether a mosaic can be built given the image count and the
    requested number of rows and columns. Raise a `ValueError` if it cannot be
    built.

    Parameters
    ----------
    img_count : int
        Image count to be arranged in the mosaic.
    rows : int
        Row count.
    cols : int
        Column count.
    """

    cell_count = rows * cols

    if img_count < cell_count:
        raise ValueError(
            f"Less slices than cells requested.\nImage count: {img_count}; "
            f"Cell count: {cell_count} (rows: {rows}; cols: {cols}).\n"
            "Please provide an appropriate value for the rows, cols for the "
            "slice count.")
    elif img_count > cell_count:
        raise ValueError(
            f"More slices than cells requested.\nImage count: {img_count}; "
            f"Cell count: {cell_count} (rows: {rows}; cols: {cols}).\n"
            "Please provide an appropriate value for the rows, cols for the "
            "slice count.")


def rgb2gray4pil(rgb_arr):
    """Convert an RGB array to grayscale and convert to `uint8` so that it can
    be appropriately handled by `PIL`.

    Parameters
    ----------
    rgb_arr : ndarray
        RGB value data.

    Returns
    -------
    Grayscale `unit8` data.
    """

    def _rgb2gray(rgb):
        img = Image.fromarray(np.uint8(rgb * 255)).convert("L")
        return np.array(img)

    # Convert from RGB to grayscale
    gray_arr = _rgb2gray(rgb_arr)

    # Relocate overflow values to the dynamic range
    return (gray_arr * 255).astype("uint8")


def create_image_from_scene(scene, size, mode=None, cmap_name=None):
    """Create a `PIL.Image` from the scene data.

    Parameters
    ----------
    scene : ndarray
        Scene data.
    size : array-like
        Image size (pixels) (width, height).
    mode : str, optional
        Type and depth of a pixel in the `Pillow` image.
    cmap_name : str, optional
        Colormap name.

    Returns
    -------
    image : PIL.Image
        Image.
    """

    _arr = scene
    if cmap_name:
        # Apply the colormap
        cmap = get_colormap(cmap_name)
        # data returned by cmap is normalized to the [0,1] range: scale to the
        # [0, 255] range and convert to uint8 for Pillow
        _arr = (cmap(_arr) * 255).astype("uint8")

    # Need to flip the array due to some bug in the FURY image buffer. Might be
    # solved in newer versions of the package.
    image = Image.fromarray(_arr, mode=mode).transpose(Image.FLIP_TOP_BOTTOM)

    return image.resize(size, Image.LANCZOS)


def create_mask_from_scene(scene, size):
    """Create a binary `PIL.Image` from the scene data.

    Parameters
    ----------
    scene : ndarray
        Scene data.
    size : array-like
        Image size (pixels) (width, height).

    Returns
    -------
    image : PIL.Image
        Image.
    """

    _bin_arr = scene > 0
    _bin_arr = rgb2gray4pil(_bin_arr) * 255
    image = create_image_from_scene(_bin_arr, size)

    return image


def draw_scene_at_pos(
    canvas,
    scene,
    size,
    left_pos,
    top_pos,
    transparency=None,
    labelmap_overlay=None,
    labelmap_overlay_alpha=0.7,
    mask_overlay=None,
    mask_overlay_alpha=0.7,
    mask_overlay_color=None,
    vol_cmap_name=None,
    labelmap_cmap_name=None,
):
    """Draw a scene in the given target image at the specified position.

    Parameters
    ----------
    canvas : PIL.Image
        Target image.
    scene : ndarray
        Scene data to be drawn.
    size : array-like
        Image size (pixels) (width, height).
    left_pos : int
        Left position (pixels).
    top_pos : int
        Top position (pixels).
    transparency : ndarray, optional
        Transparency mask.
    labelmap_overlay : ndarray
        Labelmap overlay scene data to be drawn.
    mask_overlay : ndarray
        Mask overlay scene data to be drawn.
    mask_overlay_alpha : float
        Alpha value for mask overlay in range [0, 1].
    mask_overlay_color : list, optional
        Color for the mask overlay as a list of 3 integers in range [0, 255].
    vol_cmap_name : str, optional
        Colormap name for the image scene data.
    labelmap_cmap_name : str, optional
        Colormap name for the labelmap overlay scene data.
    """

    image = create_image_from_scene(scene, size, cmap_name=vol_cmap_name)

    trans_img = None
    if transparency is not None:
        trans_img = create_image_from_scene(transparency, size, mode="L")

    canvas.paste(image, (left_pos, top_pos), mask=trans_img)

    # Draw the labelmap overlay image if any
    if labelmap_overlay is not None:
        labelmap_img = create_image_from_scene(
            labelmap_overlay, size, cmap_name=labelmap_cmap_name
        )

        # Create transparency mask over the labelmap overlay image
        label_mask = labelmap_overlay > 0
        label_transparency = create_image_from_scene(
            (label_mask * labelmap_overlay_alpha * 255.).astype(np.uint8),
            size).convert("L")

        canvas.paste(labelmap_img, (left_pos, top_pos), mask=label_transparency)

    # Draw the mask overlay image if any
    if mask_overlay is not None:
        if mask_overlay_color is None:
            # Get a list of distinguishable colors if None are supplied
            mask_overlay_color = distinguishable_colormap(
                nb_colors=len(mask_overlay))

        for img, color in zip(mask_overlay, mask_overlay_color):
            overlay_img = create_image_from_scene(
                (img * color).astype(np.uint8), size, "RGB")

            # Create transparency mask over the mask overlay image
            overlay_trans = create_image_from_scene(
                (img * mask_overlay_alpha).astype(np.uint8), size).convert("L")

            canvas.paste(overlay_img, (left_pos, top_pos), mask=overlay_trans)


def compute_canvas_size(
    cell_width,
    cell_height,
    overlap_horiz,
    overlap_vert,
    rows,
    cols,
):
    """Compute the size of a canvas with the given number of rows and columns,
    and the requested cell size and overlap values.

    Parameters
    ----------
    cell_width : int
        Cell width (pixels).
    cell_height : int
        Cell height (pixels).
    overlap_horiz : int
        Horizontal overlap (pixels).
    overlap_vert : int
        Vertical overlap (pixels).
    rows : int
        Row count.
    cols : int
        Column count.
    """

    def _compute_canvas_length(line_count, cell_length, overlap):
        return (line_count - 1) * (cell_length - overlap) + cell_length

    width = _compute_canvas_length(cols, cell_width, overlap_horiz)
    height = _compute_canvas_length(rows, cell_height, overlap_vert)

    return width, height


def create_canvas(
    cell_width,
    cell_height,
    overlap_horiz,
    overlap_vert,
    rows,
    cols,
):
    """Create a canvas for given number of rows and columns, and the requested
     cell size and overlap values.

    Parameters
    ----------
    cell_width : int
        Cell width (pixels).
    cell_height : int
        Cell height (pixels).
    overlap_horiz : int
        Horizontal overlap (pixels).
    overlap_vert : int
        Vertical overlap (pixels).
    rows : int
        Row count.
    cols : int
        Column count.
    """

    width, height = compute_canvas_size(
        cell_width, cell_height, overlap_horiz, overlap_vert, rows, cols
    )
    mosaic = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    return mosaic


def compute_cell_topleft_pos(idx, cols, offset_h, offset_v):
    """Compute the top-left position of a cell to be drawn in a mosaic.

    Parameters
    ----------
    idx : int
       Cell index in the mosaic.
    cols : int
        Column count.
    offset_h :
        Horizontal offset (pixels).
    offset_v :
        Vertical offset (pixels).
    """

    row_idx = int(np.floor(idx / cols))
    top_pos = row_idx * offset_v
    col_idx = idx % cols
    left_pos = col_idx * offset_h

    return top_pos, left_pos


def annotate_scene(mosaic, slice_number, display_slice_number, display_lr):
    font_size = mosaic.width // 10
    font = ImageFont.truetype(
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf', font_size)

    stroke, padding = max(mosaic.width // 200, 1), mosaic.width // 100
    img = ImageDraw.Draw(mosaic)

    if display_slice_number:
        img.text(
            (padding, padding), "{}".format(slice_number), (255,255,255),
            font=font, stroke_width=stroke, stroke_fill=(0, 0, 0)
        )

    if display_lr:
        l_text, r_text = "L", "R"
        if display_lr < 0:
            l_text, r_text = r_text, l_text

        img.text(
            (padding, mosaic.height // 2), l_text, (255,255,255),
            font=font, anchor="lm", stroke_width=stroke, stroke_fill=(0, 0, 0)
        )
        img.text(
            (mosaic.width - padding, mosaic.height // 2), r_text, (255,255,255),
            font=font, anchor="rm", stroke_width=stroke, stroke_fill=(0, 0, 0)
        )


def compose_mosaic(
    img_scene_container,
    cell_size,
    rows,
    cols,
    slice_numbers,
    overlap_factor=None,
    transparency_scene_container=None,
    labelmap_scene_container=None,
    labelmap_overlay_alpha=0.7,
    mask_overlay_scene_container=None,
    mask_overlay_alpha=0.7,
    mask_overlay_color=None,
    vol_cmap_name=None,
    labelmap_cmap_name=None,
    display_slice_number=False,
    display_lr=False
):
    """Create the mosaic canvas for given number of rows and columns, and the
    requested cell size and overlap values.

    Parameters
    ----------
    img_scene_container : list
        Image scene data container.
    cell_size : array-like
        Cell size (pixels) (width, height).
    rows : int
        Row count.
    cols : int
        Column count.
    overlap_factor : array-like
        Overlap factor (horizontal, vertical).
    transparency_scene_container : list, optional
        Transaprency scene data container.
    labelmap_scene_container : list, optional
        Labelmap scene data container.
    mask_overlay_scene_container : list, optional
        Mask overlay scene data container.
    mask_overlay_alpha : float, optional
        Alpha value for mask overlay in range [0, 1].
    mask_overlay_color : list, optional
        Color for the mask overlay as a list of 3 integers in range [0, 255].
    vol_cmap_name : str, optional
        Colormap name for the image scene data.
    labelmap_cmap_name : str, optional
        Colormap name for the labelmap scene data.
    display_slice_number : bool, optional
        If true, displays the slice number in the upper left corner.
    display_lr : bool or int, optional
        If 1 or -1, identifies the left and right sides on the image. -1 flips 
        left and right positions.
    """

    def _compute_overlap_length(length, _overlap):
        return round(length * _overlap)

    cell_width = cell_size[0]
    cell_height = cell_size[1]

    overlap_h = overlap_v = 0
    if overlap_factor is not None:
        overlap_h = _compute_overlap_length(cell_width, overlap_factor[0])
        overlap_v = _compute_overlap_length(cell_width, overlap_factor[1])

    mosaic = create_canvas(*cell_size, overlap_h, overlap_v, rows, cols)

    offset_h = cell_width - overlap_h
    offset_v = cell_height - overlap_v
    from itertools import zip_longest
    for idx, (img_arr, trans_arr, labelmap_arr, mask_overlay_arr, slice_number) in enumerate(
            list(zip_longest(
                img_scene_container,
                transparency_scene_container,
                labelmap_scene_container,
                mask_overlay_scene_container,
                slice_numbers,
                fillvalue=tuple()))
    ):

        # Compute the mosaic cell position
        top_pos, left_pos = compute_cell_topleft_pos(
            idx, cols, offset_h, offset_v
        )

        # Convert the scene data to grayscale and adjust for handling with
        # Pillow
        _img_arr = rgb2gray4pil(img_arr)

        _trans_arr = None
        if len(trans_arr):
            _trans_arr = rgb2gray4pil(trans_arr)

        _labelmap_arr = None
        if len(labelmap_arr):
            _labelmap_arr = rgb2gray4pil(labelmap_arr)

        _mask_overlay_arr = None
        if len(mask_overlay_arr):
            _mask_overlay_arr = mask_overlay_arr

        # Draw the image (and labelmap overlay, if any) in the cell
        draw_scene_at_pos(
            mosaic,
            _img_arr,
            (cell_width, cell_height),
            left_pos,
            top_pos,
            transparency=_trans_arr,
            labelmap_overlay=_labelmap_arr,
            labelmap_overlay_alpha=labelmap_overlay_alpha,
            mask_overlay=_mask_overlay_arr,
            mask_overlay_alpha=mask_overlay_alpha,
            mask_overlay_color=mask_overlay_color,
            vol_cmap_name=vol_cmap_name,
            labelmap_cmap_name=labelmap_cmap_name,
        )

        if display_slice_number or display_lr:
            annotate_scene(
                mosaic, slice_number, display_slice_number, display_lr)

    return mosaic
