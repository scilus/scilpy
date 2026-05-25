# -*- coding: utf-8 -*-

import numpy as np
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.direction.peaks import peak_directions
from dipy.reconst.shm import sh_to_sf_matrix


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

    # If all triplets have the same norm, it is likely peaks, otherwise SH.
    if last_dim % 3 == 0:
        norm = np.linalg.norm(data_nz.reshape(-1, 3), axis=-1)
        if np.all(np.isclose(norm, norm[0])):
            return True

    # If the max is in the first triplet but not at index 0, it's likely Peaks.
    # Smoothed SH almost always has max at index 0
    argmax_indices = np.argmax(np.abs(data_nz), axis=-1)
    if last_dim % 3 == 0 and \
            np.mean(np.logical_or(argmax_indices == 1,
                                  argmax_indices == 2)) > 0.1:
        return True

    # Exact zeros. SH almost never has exact zeros in real data.
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
    Only computes SF for voxels where data is non-zero (or in mask) to save
    RAM.

    This information can be used to compute a global threshold for SF
    amplitude, which is often used to filter out spurious peaks in fODF.

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

    In SF obtained from fODF, the amplitude of the lobes corresponds to the
    strength of the diffusion signal in those directions. Thresholding these
    amplitudes is a common practice to filter out spurious peaks.

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
    postprocess_mask : bool, optional
        Whether to postprocess the mask to keep only the largest component.

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
        if data.ndim == 5:
            if data.shape[-1] != 3:
                raise ValueError("5D peaks input must have 3 "
                                 "as last dimension.")
            peaks = data
        elif data.ndim == 4:
            npeaks = data.shape[-1] // 3
            peaks = data.reshape(data.shape[:3] + (npeaks, 3))
        else:
            raise ValueError("Peaks input must be 4D or 5D.")

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
        if relative_factor < 0 or relative_factor > 1:
            raise ValueError("relative_factor must be between 0 and 1.")
        threshold = relative_factor * global_max

    if global_max == 0:
        mask = np.zeros(max_amp.shape, dtype=bool)
    else:
        mask = max_amp >= threshold

    if postprocess_mask and np.any(mask):
        import scipy.ndimage as ndi
        # Postprocess to label all elements and count voxels for each label
        labels = ndi.label(mask)[0]
        label_counts = np.bincount(labels.ravel())

        # Guard against empty label_counts[1:]
        if len(label_counts) > 1:
            # Find the largest connected component (excluding background)
            # +1 to skip background
            largest_label = np.argmax(label_counts[1:]) + 1
            # Create a mask for the largest connected component
            mask = labels == largest_label
            inverted_mask = ~mask

            # Remove isolated voxels in the inverted mask (holes in main mask)
            labels_inverted = ndi.label(inverted_mask)[0]
            label_counts_inverted = np.bincount(labels_inverted.ravel())
            for label, count in enumerate(label_counts_inverted):
                if label == 0:
                    continue  # Skip background
                if count < 100:  # Threshold for filling holes
                    mask[labels_inverted == label] = True

    return mask, global_max, threshold
