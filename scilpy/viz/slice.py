# -*- coding: utf-8 -*-

from dipy.reconst.shm import sh_to_sf
from fury import actor
import numpy as np

from scilpy.reconst.bingham import bingham_to_sf
from scilpy.viz.backends.fury import (create_contours_actor,
                                      create_odf_actors,
                                      create_peaks_actor,
                                      set_display_extent)
from scilpy.viz.backends.vtk import contours_from_data
from scilpy.viz.color import generate_n_colors, lut_from_matplotlib_name
from scilpy.viz.utils import affine_from_offset


def create_texture_slicer(texture, orientation, slice_index, mask=None,
                          value_range=None, opacity=1.0, offset=0.5,
                          lut=None, interpolation='nearest'):
    """
    Create a texture displayed at a given offset (in the given orientation)
    from the origin of the grid.

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
    opacity : float
        The opacity of the texture image. Opacity of 0.0 means transparent and
        1.0 is completely visible. Defaults to 1.0.
    offset : float
        The offset of the texture image. Defaults to 0.5.
    lut : str, vtkLookupTable, optional
        Either a vtk lookup table or a matplotlib name for one.
    interpolation : str
        Interpolation mode for the texture image. Choices are nearest or
        linear. Defaults to nearest.

    Returns
    -------
    slicer_actor : actor.slicer
        Fury object containing the texture information.
    """

    affine = affine_from_offset(orientation, offset)

    if mask is not None:
        texture[np.where(mask == 0)] = 0

    if isinstance(lut, str):
        _vl = value_range
        if _vl is None:
            _vl = (texture.min(), texture.max())

        lut = lut_from_matplotlib_name(lut, _vl)

    slicer_actor = actor.slicer(texture, value_range=value_range,
                                affine=affine, opacity=opacity,
                                lookup_colormap=lut,
                                interpolation=interpolation)

    set_display_extent(slicer_actor, orientation, texture.shape, slice_index)

    return slicer_actor


def create_contours_slicer(data, contour_values, orientation, slice_index,
                           smoothing_radius=0., opacity=1., linewidth=3.,
                           color=[255, 0, 0]):
    """
    Create an isocontour slicer at specifed contours values.

    Parameters
    ----------
    data : np.ndarray
        Data from which to extract contours (mask, binary image, labels).
    contour_values : list
        Values at which to extract isocontours.
    orientation : str
        Name of the axis to visualize. Choices are axial, coronal and sagittal.
    slice_index : int
        Index of the slice to visualize along the chosen orientation.
    smoothing_radius : float
        Pre-smoothing to apply to the image before
        computing the contour (in pixels).
    opacity: float
        Opacity of the contour.
    linewidth : float
        Thickness of the contour line.
    color : tuple, list of int
        Color of the contour in RGB [0, 255].

    Returns
    -------
    contours_slicer : actor.slicer
        Fury object containing the contours information.
    """

    data = np.rot90(data.take([slice_index], orientation).squeeze())
    contours_polydata = contours_from_data(data, contour_values,
                                           smoothing_radius)
    contours_slicer = create_contours_actor(contours_polydata, opacity,
                                            linewidth, color)

    # Equivalent of set_display_extent for polydata actors
    position = [0, 0, 0]
    position[orientation] = slice_index

    if orientation == 0:
        contours_slicer.SetOrientation(90, 0, 90)
    elif orientation == 1:
        contours_slicer.SetOrientation(90, 0, 0)

    contours_slicer.SetPosition(*position)

    return contours_slicer


def create_peaks_slicer(data, orientation, slice_index, peak_values=None,
                        mask=None, color=None, peaks_width=1.0,
                        opacity=1.0, symmetric=False):
    """
    Create a peaks slicer actor rendering a slice of the input peaks.

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
    peaks_width : float
        Width of peaks segments. Defaults to 1.0.
    opacity : float
        Opacity of the peaks. Defaults to 1.0.
    symmetric : bool
        If True, peaks are drawn for both peaks_dirs and -peaks_dirs. Else,
        peaks are only drawn for directions given by peaks_dirs. Defaults to
        False.

    Returns
    -------
    slicer_actor : actor.peak_slicer
        Fury object containing the peaks information.
    """

    # Reshape peaks volume to XxYxZxNx3
    data = data.reshape(data.shape[:3] + (-1, 3))
    norm = np.linalg.norm(data, axis=-1)

    # Only send non-empty data slices to render
    zero_norms = np.sum(norm.reshape((-1, norm.shape[-1])), axis=0) == 0

    if zero_norms.all():
        raise ValueError('Peak slicer received an empty volume to render.')

    data = data[..., ~zero_norms, :]
    norm = norm[..., ~zero_norms]

    # Normalize input data
    data[norm > 0] /= norm[norm > 0].reshape((-1, 1))

    # Instantiate peaks slicer
    peaks_slicer = create_peaks_actor(data, mask, opacity=opacity,
                                      linewidth=peaks_width, color=color,
                                      lut_values=peak_values,
                                      symmetric=symmetric)

    set_display_extent(peaks_slicer, orientation, data.shape, slice_index)

    return peaks_slicer


def create_odf_slicer(sh_fodf, orientation, slice_index, sphere, sh_order,
                      sh_basis, full_basis, scale, sh_variance=None,
                      mask=None, nb_subdivide=None, radial_scale=False,
                      norm=False, colormap=None, variance_k=1,
                      variance_color=(255, 255, 255), is_legacy=True):
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
    sphere: DIPY Sphere
        Sphere used for visualization.
    sh_order : int
        Maximum spherical harmonics order.
    sh_basis : str
        Type of basis for the spherical harmonics.
    full_basis : bool
        Boolean indicating if the basis is full or not.
    scale : float
        Scaling factor for FODF.
    sh_variance : np.ndarray, optional
        Spherical harmonics of the variance fODF data.
    mask : np.ndarray, optional
        Only the data inside the mask will be displayed. Defaults to None.
    nb_subdivide : int, optional
        Number of subdivisions for given sphere. If None, uses the given sphere
        as is.
    radial_scale : bool
        If True, enables radial scale for ODF slicer.
    norm : bool
        If True, enables normalization of ODF slicer.
    colormap : str, optional
        Colormap for the ODF slicer. If None, a RGB colormap is used.
    variance_k : float
        Factor that multiplies sqrt(variance).
    variance_color : tuple, optional
        Color of the variance fODF data, in RGB.

    Returns
    -------
    odf_actor : actor.odf_slicer
        Fury object containing the odf information.
    var_actor : actor.odf_slicer
        Fury object containing the odf variance information.
    """

    # Subdivide the spheres if nb_subdivide is provided
    if nb_subdivide is not None:
        sphere = sphere.subdivide(nb_subdivide)

    fodf = sh_to_sf(sh_fodf, sphere, sh_order, sh_basis,
                    full_basis=full_basis, legacy=is_legacy)

    fodf_var = None
    if sh_variance is not None:
        fodf_var = sh_to_sf(sh_variance, sphere, sh_order, sh_basis,
                            full_basis=full_basis, legacy=is_legacy)

    odf_actor, var_actor = create_odf_actors(fodf, sphere, scale, fodf_var,
                                             mask, radial_scale,
                                             norm, colormap,
                                             variance_k, variance_color)

    set_display_extent(odf_actor, orientation, sh_fodf.shape[:3], slice_index)
    if sh_variance is not None:
        set_display_extent(var_actor, orientation,
                           sh_fodf.shape[:3], slice_index)

    return odf_actor, var_actor


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
    color_per_lobe: bool
        If true, each Bingham distribution is colored using a disting color.
        Else, Bingham distributions are colored by their orientation.

    Return
    ------
    actors: list of fury odf_slicer actors
        ODF slicer actors representing the Bingham distributions.
    """
    shape = data.shape
    nb_lobes = shape[-2]
    colors = [c * 255 for c in generate_n_colors(nb_lobes)]

    # lmax norm for normalization
    lmaxnorm = np.max(np.abs(data[..., 0]), axis=-1)
    bingham_sf = bingham_to_sf(data, sphere.vertices)

    actors = []
    for nn in range(nb_lobes):
        sf = bingham_sf[..., nn, :]
        sf[lmaxnorm > 0] /= lmaxnorm[lmaxnorm > 0][:, None]
        color = colors[nn] if color_per_lobe else None

        odf_actor, _ = create_odf_actors(sf, sphere, 0.5, colormap=color,
                                         radial_scale=True)

        set_display_extent(odf_actor, orientation, shape[:3], slice_index)
        actors.append(odf_actor)

    return actors
