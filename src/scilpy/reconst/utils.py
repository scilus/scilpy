# -*- coding: utf-8 -*-

from dipy.direction.peaks import peak_directions
import numpy as np


def find_order_from_nb_coeff(data):
    if isinstance(data, np.ndarray):
        shape = data.shape
    else:
        shape = data
    return int((-3 + np.sqrt(1 + 8 * shape[-1])) / 2)


def get_sh_order_and_fullness(ncoeffs):
    """
    Get the order of the SH basis from the number of SH coefficients
    as well as a boolean indicating if the basis is full.
    """
    # the two curves (sym and full) intersect at ncoeffs = 1, in what
    # case both bases correspond to order 1.
    sym_order = (-3.0 + np.sqrt(1.0 + 8.0 * ncoeffs)) / 2.0
    if sym_order.is_integer():
        return sym_order, False
    full_order = np.sqrt(ncoeffs) - 1.0
    if full_order.is_integer():
        return full_order, True
    raise ValueError('Invalid number of coefficients for SH basis.')


def get_maximas(data, sphere, b_matrix, threshold, absolute_threshold,
                min_separation_angle=25):
    spherical_func = np.dot(data, b_matrix.T)
    spherical_func[np.nonzero(spherical_func < absolute_threshold)] = 0.
    return peak_directions(
        spherical_func, sphere, threshold, min_separation_angle)


def get_sphere_neighbours(sphere, max_angle):
    """
    Get a matrix of neighbours for each direction on the sphere, within
    the min_separation_angle.

    min_separation_angle: float
        Maximum angle in radians defining the neighbourhood
        of each direction.

    Return
    ------
    neighbours: ndarray
        Neighbour directions for each direction on the sphere.
    """
    xs = sphere.vertices[:, 0]
    ys = sphere.vertices[:, 1]
    zs = sphere.vertices[:, 2]
    scalar_prods = (np.outer(xs, xs) + np.outer(ys, ys) +
                    np.outer(zs, zs))
    neighbours = scalar_prods >= np.cos(max_angle)
    return neighbours


def is_data_peaks(img_data):
    """
    Heuristic to find out if the input are peaks or fodf.
    fodf are always around 0.15 and peaks around 0.75.
    Peaks have more zero values than fodf. The first value of fodf is
    usually the highest.

    Parameters
    ----------
    img_data : np.ndarray
        4D image data where the last dimension contains directional info.

    Returns
    -------
    is_peaks : bool
        True if data is likely peaks, False if likely fODF (SH).
    """
    # Sum of absolute values to detect non-zero voxels correctly
    non_zeros_mask = np.any(np.abs(img_data) > 0, axis=-1)
    non_zeros_count = np.count_nonzero(non_zeros_mask)
    if non_zeros_count == 0:
        return False

    # Filter only non-zero voxels for more accurate argmax
    # Peaks usually have non-zero indices for max amplitude
    # SH (fODF) usually has the first coefficient as the highest (DC component)
    if img_data.shape[-1] == 1:
        return False

    non_first_val_count = np.count_nonzero(np.argmax(img_data[non_zeros_mask],
                                                     axis=-1))
    return non_first_val_count / non_zeros_count > 0.5


def compute_sf_threshold_mask(data, sphere, relative_factor=None,
                              absolute_threshold=None,
                              basis='descoteaux07',
                              is_legacy=True, nbr_processes=None):
    """
    Compute a binary mask based on a global SF threshold.

    Parameters
    ----------
    data : np.ndarray
        ODF data (SH or Peaks).
    sphere : dipy.core.sphere.Sphere
        Sphere for SF sampling (for SH).
    relative_factor : float, optional
        Factor between 0 and 1. Threshold is factor * global_max_sf.
    absolute_threshold : float, optional
        Absolute threshold on SF amplitude.
    basis : str
        SH basis.
    is_legacy : bool
        If True, use legacy SH basis.
    nbr_processes : int
        Number of processes for parallel computation.

    Returns
    -------
    mask : np.ndarray
        Binary mask.
    global_max : float
        Global maximum SF amplitude (useful if relative_factor was used).
    threshold : float
        Computed threshold value.
    """
    if relative_factor is None and absolute_threshold is None:
        raise ValueError("Either relative_factor or absolute_threshold "
                         "must be provided.")

    is_peaks = is_data_peaks(data)
    if is_peaks:
        # Data is peaks: [x,y,z, npeaks*3]
        npeaks = data.shape[-1] // 3
        # Reshape to [x,y,z, npeaks, 3]
        peaks = data.reshape(data.shape[:3] + (npeaks, 3))
        # Norms: [x,y,z, npeaks]
        norms = np.linalg.norm(peaks, axis=-1)
        # Max per voxel: [x,y,z]
        max_sf = np.max(norms, axis=-1)
    else:
        # Data is SH
        from scilpy.reconst.sh import peaks_from_sh
        # We need a mask to avoid computing on empty voxels and to help
        # peaks_from_sh which might have issues with all-zero voxels if
        # not handled.
        mask_data = np.sum(np.abs(data), axis=-1) > 0
        max_sf = np.zeros(data.shape[:3])
        if np.any(mask_data):
            # npeaks=1 is enough to find the maximum on the sphere
            _, peak_values, _ = peaks_from_sh(data.astype(np.float32),
                                              sphere, mask=mask_data,
                                              relative_peak_threshold=0.0,
                                              npeaks=1,
                                              sh_basis_type=basis,
                                              is_legacy=is_legacy,
                                              nbr_processes=nbr_processes)
            max_sf[mask_data] = peak_values[mask_data, 0]

    global_max = np.max(max_sf) if max_sf.size > 0 else 0.0

    if absolute_threshold is not None:
        threshold = absolute_threshold
    else:
        threshold = relative_factor * global_max

    mask = max_sf >= threshold
    return mask, global_max, threshold
