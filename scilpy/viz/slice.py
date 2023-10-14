from dipy.reconst.shm import sh_to_sf, sh_to_sf_matrix
from fury import actor
from fury.colormap import distinguishable_colormap

import numpy as np
from scilpy.reconst.bingham import bingham_to_sf
from scilpy.viz.backends.fury import set_display_extent
from scilpy.viz.utils import affine_from_offset


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


def create_texture_slicer(texture, orientation, slice_index, mask=None,
                          value_range=None, opacity=1.0, offset=0.5,
                          interpolation='nearest'):
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
    affine = affine_from_offset(orientation, offset)

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
    Create a peaks slicer actor rendering a slice of the input peaks

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