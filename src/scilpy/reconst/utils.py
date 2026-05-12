# -*- coding: utf-8 -*-

from dipy.direction.peaks import peak_directions
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes


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
        spherical_func, sphere,
        relative_peak_threshold=threshold,
        min_separation_angle=min_separation_angle)


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
    last_dim = img_data.shape[-1]
    if last_dim == 3:
        return True

    # Sum of absolute values to detect non-zero voxels correctly
    non_zeros_mask = np.any(np.abs(img_data) > 0, axis=-1)
    if not np.count_nonzero(non_zeros_mask):
        return False

    try:
        order, full = get_sh_order_and_fullness(last_dim)
        # Symmetric SH must be even order
        if not full and order % 2 != 0:
            return False
    except ValueError:
        # If not a valid SH number of coefficients, and not 3,
        # it might be something else, but if it's a multiple of 3
        # it's likely Peaks.
        if last_dim % 3 == 0:
            return True
        return False

    data_nz = img_data[non_zeros_mask]

    # Heuristic 1: Argmax distribution.
    # In Peaks (sorted), the max is always in the first triplet (index 0, 1, 2).
    # In SH, the max can be anywhere (DC at 0, or higher orders for sharp ODFs)
    argmax_indices = np.argmax(np.abs(data_nz), axis=-1)

    # If the max is frequently outside the first triplet, it's likely SH
    if np.mean(argmax_indices > 2) > 0.1:
        return False

    # If the max is in the first triplet but not at index 0, it's likely Peaks.
    # Smoothed SH almost always has max at index 0
    if np.mean(np.logical_or(argmax_indices == 1, argmax_indices == 2)) > 0.1:
        return True

    # Heuristic 2: Exact zeros. SH almost never has exact zeros in real data.
    # Peaks often have exact zeros for unused lobes
    zero_ratio = np.mean(data_nz == 0)
    if zero_ratio > 0.05:
        return True

    # Default to SH
    return False

def compute_max_sf_amplitude(data, sh_basis, is_legacy,
                             sphere_name='repulsion100', mask=None):
    """
    Compute the maximum SF amplitude for each voxel.
    Only computes SF for voxels where data is non-zero (or in mask) to save RAM.

    Parameters
    ----------
    data : np.ndarray
        ODF data (SH).
    sh_basis : str
        SH basis ('tournier07' or 'descoteaux07').
    is_legacy : bool
        Whether the SH basis is legacy.
    sphere_name : str or dipy.core.sphere.Sphere, optional
        Sphere name for SF conversion or Sphere object.
    mask : np.ndarray, optional
        Binary mask. If provided, only voxels in mask are computed.

    Returns
    -------
    max_sf : np.ndarray
        Maximum SF amplitude per voxel.
    """
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix
    from dipy.core.sphere import Sphere

    if mask is None:
        mask = np.any(data, axis=-1)

    order = find_order_from_nb_coeff(data)
    if isinstance(sphere_name, (Sphere,)):
        sphere = sphere_name
    else:
        sphere = get_sphere(name=sphere_name)

    b_matrix, _ = sh_to_sf_matrix(sphere, sh_order_max=order,
                                  basis_type=sh_basis, legacy=is_legacy)

    max_sf = np.zeros(data.shape[:-1], dtype=np.float32)
    if np.any(mask):
        # Vectorized SF computation for masked voxels
        sf = np.dot(data[mask], b_matrix)
        max_sf[mask] = np.max(sf, axis=-1)

    return max_sf


def compute_sf_threshold_mask(data, sphere_name='repulsion100',
                              relative_factor=None,
                              absolute_threshold=None,
                              sh_basis='descoteaux07',
                              is_legacy=True, postprocess_mask=True):
    """
    Compute a binary mask based on a global SF amplitude threshold.

    Parameters
    ----------
    data : np.ndarray
        ODF data (SH or Peaks).
    sphere_name : str or dipy.core.sphere.Sphere, optional
        Sphere name for SF conversion or Sphere object.
    relative_factor : float, optional
        Factor between 0 and 1. Threshold is factor * global_max_sf.
    absolute_threshold : float, optional
        Absolute threshold on SF amplitude.
    sh_basis : str, optional
        SH basis ('tournier07' or 'descoteaux07').
    is_legacy : bool, optional
        Whether the SH basis is legacy.

    Returns
    -------
    mask : np.ndarray
        Binary mask.
    global_max : float
        Global maximum SF amplitude.
    threshold : float
        Computed threshold value.
    """
    if relative_factor is None and absolute_threshold is None:
        raise ValueError("Either relative_factor or absolute_threshold "
                         "must be provided.")

    is_peaks = is_data_peaks(data)
    if is_peaks:
        npeaks = data.shape[-1] // 3
        peaks = data.reshape(data.shape[:3] + (npeaks, 3))
        norms = np.linalg.norm(peaks, axis=-1)
        # maximum amplitude/norm across peaks
        max_amp = np.max(norms, axis=-1)
    else:
        max_amp = compute_max_sf_amplitude(data, sh_basis, is_legacy,
                                           sphere_name=sphere_name)

    global_max = np.max(max_amp)

    if absolute_threshold is not None:
        threshold = absolute_threshold
    else:
        threshold = relative_factor * global_max

    mask = max_amp >= threshold

    if postprocess_mask:
        # Post-process to remove single voxels and fill single voxel holes
        mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
        # Invert the image to fill holes in the mask, then invert back
        mask = np.logical_not(binary_fill_holes(np.logical_not(mask)))

    return mask, global_max, threshold
