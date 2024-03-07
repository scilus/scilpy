# -*- coding: utf-8 -*-
import itertools
import logging
import multiprocessing
import numpy as np

from dipy.core.sphere import Sphere
from dipy.direction.peaks import peak_directions
from dipy.reconst.odf import gfa
from dipy.reconst.shm import (sh_to_sf_matrix, order_from_ncoef, sf_to_sh,
                              sph_harm_ind_list)

from scilpy.gradients.bvec_bval_tools import (identify_shells,
                                              is_normalized_bvecs,
                                              normalize_bvecs,
                                              DEFAULT_B0_THRESHOLD)
from scilpy.dwi.operations import compute_dwi_attenuation


def verify_data_vs_sh_order(data, sh_order):
    """
    Raises a warning if the dwi data shape is not enough for the chosen
    sh_order.

    Parameters
    ----------
    data: np.ndarray
        Diffusion signal as weighted images (4D).
    sh_order: int
        SH order to fit, by default 4.
    """
    if data.shape[-1] < (sh_order + 1) * (sh_order + 2) / 2:
        logging.warning(
            'We recommend having at least {} unique DWIs volumes, but you '
            'currently have {} volumes. Try lowering the parameter --sh_order '
            'in case of non convergence.'.format(
                (sh_order + 1) * (sh_order + 2) / 2, data.shape[-1]))


def compute_sh_coefficients(dwi, gradient_table,
                            b0_threshold=DEFAULT_B0_THRESHOLD, sh_order=4,
                            basis_type='descoteaux07', smooth=0.006,
                            use_attenuation=False, mask=None, sphere=None,
                            is_legacy=True):
    """Fit a diffusion signal with spherical harmonics coefficients.
    Data must come from a single shell acquisition.

    Parameters
    ----------
    dwi : nib.Nifti1Image object
        Diffusion signal as weighted images (4D).
    gradient_table : GradientTable
        Dipy object that contains all bvals and bvecs.
    b0_threshold: float
        Threshold for the b0 values. Used to validate that the data contains
        single shell signal.
    sh_order : int, optional
        SH order to fit, by default 4.
    basis_type: str
        Either 'tournier07' or 'descoteaux07'
    smooth : float, optional
        Lambda-regularization coefficient in the SH fit, by default 0.006.
    use_attenuation: bool, optional
        If true, we will use DWI attenuation. [False]
    mask: nib.Nifti1Image object, optional
        Binary mask. Only data inside the mask will be used for computations
        and reconstruction.
    sphere: Sphere
        Dipy object. If not provided, will use Sphere(xyz=bvecs).
    is_legacy : bool, optional
        Whether or not the SH basis is in its legacy form.

    Returns
    -------
    sh_coeffs : np.ndarray with shape (X, Y, Z, #coeffs)
        Spherical harmonics coefficients at every voxel. The actual number
        of coefficients depends on `sh_order`.
    """

    # Extracting infos
    b0_mask = gradient_table.b0s_mask
    bvecs = gradient_table.bvecs
    bvals = gradient_table.bvals

    # Checks
    if not is_normalized_bvecs(bvecs):
        logging.warning("Your b-vectors do not seem normalized...")
        bvecs = normalize_bvecs(bvecs)

    # Ensure that this is on a single shell.
    shell_values, _ = identify_shells(bvals)
    shell_values.sort()
    if shell_values.shape[0] != 2 or shell_values[0] > b0_threshold:
        raise ValueError("Can only work on single shell signals.")

    # Keeping b0-based infos
    bvecs = bvecs[np.logical_not(b0_mask)]
    weights = dwi[..., np.logical_not(b0_mask)]

    # Compute attenuation using the b0.
    if use_attenuation:
        b0 = dwi[..., b0_mask].mean(axis=3)
        weights = compute_dwi_attenuation(weights, b0)

    # Get cartesian coords from bvecs
    if sphere is None:
        sphere = Sphere(xyz=bvecs)

    # Fit SH
    sh = sf_to_sh(weights, sphere, sh_order, basis_type, smooth=smooth,
                  legacy=is_legacy)

    # Apply mask
    if mask is not None:
        sh *= mask[..., None]

    return sh


def compute_rish(sh, mask=None, full_basis=False):
    """Compute the RISH (Rotationally Invariant Spherical Harmonics) features
    of the SH signal [1]. Each RISH feature map is the total energy of its
    associated order. Mathematically, it is the sum of the squared SH
    coefficients of the SH order.

    Parameters
    ----------
    sh : np.ndarray object
        Array of the SH coefficients
    mask: np.ndarray object, optional
        Binary mask. Only data inside the mask will be used for computation.
    full_basis: bool, optional
        True when coefficients are for a full SH basis.

    Returns
    -------
    rish : np.ndarray with shape (x,y,z,n_orders)
        The RISH features of the input SH, with one channel per SH order.
    orders : list(int)
        The SH order of each RISH feature in the last channel of `rish`.

    References
    ----------
    [1] Mirzaalian, Hengameh, et al. "Harmonizing diffusion MRI data across
        multiple sites and scanners." MICCAI 2015.
        https://scholar.harvard.edu/files/hengameh/files/miccai2015.pdf
    """
    # Guess SH order
    sh_order = order_from_ncoef(sh.shape[-1], full_basis=full_basis)

    # Get degree / order for all indices
    degree_ids, order_ids = sph_harm_ind_list(sh_order, full_basis=full_basis)

    # Apply mask to input
    if mask is not None:
        sh = sh * mask[..., None]

    # Get number of indices per order (e.g. for order 6, sym. : [1,5,9,13])
    step = 1 if full_basis else 2
    n_indices_per_order = np.bincount(order_ids)[::step]

    # Get start index of each order (e.g. for order 6 : [0,1,6,15])
    order_positions = np.concatenate([[0], np.cumsum(n_indices_per_order)])[:-1]

    # Get paired indices for np.add.reduceat, specifying where to reduce.
    # The last index is omitted, it is automatically replaced by len(array)-1
    # (e.g. for order 6 : [0,1, 1,6, 6,15, 15,])
    reduce_indices = np.repeat(order_positions, 2)[1:]

    # Compute the sum of squared coefficients using numpy's `reduceat`
    squared_sh = np.square(sh)
    rish = np.add.reduceat(squared_sh, reduce_indices, axis=-1)[..., ::2]

    # Apply mask
    if mask is not None:
        rish *= mask[..., None]

    orders = sorted(np.unique(order_ids))

    return rish, orders


def _peaks_from_sh_parallel(args):
    shm_coeff = args[0]
    B = args[1]
    sphere = args[2]
    relative_peak_threshold = args[3]
    absolute_threshold = args[4]
    min_separation_angle = args[5]
    npeaks = args[6]
    normalize_peaks = args[7]
    chunk_id = args[8]
    is_symmetric = args[9]

    data_shape = shm_coeff.shape[0]
    peak_dirs = np.zeros((data_shape, npeaks, 3))
    peak_values = np.zeros((data_shape, npeaks))
    peak_indices = np.zeros((data_shape, npeaks), dtype='int')
    peak_indices.fill(-1)

    for idx in range(len(shm_coeff)):
        if shm_coeff[idx].any():
            odf = np.dot(shm_coeff[idx], B)
            odf[odf < absolute_threshold] = 0.

            dirs, peaks, ind = peak_directions(odf, sphere,
                                               relative_peak_threshold,
                                               min_separation_angle,
                                               is_symmetric)

            if peaks.shape[0] != 0:
                n = min(npeaks, peaks.shape[0])

                peak_dirs[idx][:n] = dirs[:n]
                peak_indices[idx][:n] = ind[:n]
                peak_values[idx][:n] = peaks[:n]

                if normalize_peaks:
                    peak_values[idx][:n] /= peaks[0]
                    peak_dirs[idx] *= peak_values[idx][:, None]

    return chunk_id, peak_dirs, peak_values, peak_indices


def peaks_from_sh(shm_coeff, sphere, mask=None, relative_peak_threshold=0.5,
                  absolute_threshold=0, min_separation_angle=25,
                  normalize_peaks=False, npeaks=5,
                  sh_basis_type='descoteaux07', is_legacy=True,
                  nbr_processes=None, full_basis=False, is_symmetric=True):
    """Computes peaks from given spherical harmonic coefficients

    Parameters
    ----------
    shm_coeff : np.ndarray
        Spherical harmonic coefficients
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    mask : np.ndarray, optional
        If `mask` is provided, only the data inside the mask will be
        used for computations.
    relative_peak_threshold : float, optional
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
        Default: 0.5
    absolute_threshold : float, optional
        Absolute threshold on fODF amplitude. This value should be set to
        approximately 1.5 to 2 times the maximum fODF amplitude in isotropic
        voxels (ex. ventricles). `scil_fodf_max_in_ventricles.py`
        can be used to find the maximal value.
        Default: 0
    min_separation_angle : float in [0, 90], optional
        The minimum distance between directions. If two peaks are too close
        only the larger of the two is returned.
        Default: 25
    normalize_peaks : bool, optional
        If true, all peak values are calculated relative to `max(odf)`.
    npeaks : int, optional
        Maximum number of peaks found (default 5 peaks).
    sh_basis_type : str, optional
        Type of spherical harmonic basis used for `shm_coeff`. Either
        `descoteaux07` or `tournier07`.
        Default: `descoteaux07`
    is_legacy: bool, optional
        If true, this means that the input SH used a legacy basis definition
        for backward compatibility with previous ``tournier07`` and
        ``descoteaux07`` implementations.
        Default: True
    nbr_processes: int, optional
        The number of subprocesses to use.
        Default: multiprocessing.cpu_count()
    full_basis: bool, optional
        If True, SH coefficients are expressed using a full basis.
        Default: False
    is_symmetric: bool, optional
        If False, antipodal sphere directions are considered distinct.
        Default: True

    Returns
    -------
    tuple of np.ndarray
        peak_dirs, peak_values, peak_indices
    """
    sh_order = order_from_ncoef(shm_coeff.shape[-1], full_basis)
    B, _ = sh_to_sf_matrix(sphere, sh_order, sh_basis_type, full_basis,
                           legacy=is_legacy)

    data_shape = shm_coeff.shape
    if mask is None:
        mask = np.sum(shm_coeff, axis=3).astype(bool)

    nbr_processes = multiprocessing.cpu_count() if nbr_processes is None \
        or nbr_processes < 0 else nbr_processes

    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    shm_coeff = shm_coeff[mask].reshape(
        (np.count_nonzero(mask), data_shape[3]))
    chunks = np.array_split(shm_coeff, nbr_processes)
    chunk_len = np.cumsum([0] + [len(c) for c in chunks])

    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(_peaks_from_sh_parallel,
                       zip(chunks,
                           itertools.repeat(B),
                           itertools.repeat(sphere),
                           itertools.repeat(relative_peak_threshold),
                           itertools.repeat(absolute_threshold),
                           itertools.repeat(min_separation_angle),
                           itertools.repeat(npeaks),
                           itertools.repeat(normalize_peaks),
                           np.arange(len(chunks)),
                           itertools.repeat(is_symmetric)))
    pool.close()
    pool.join()

    # Re-assemble the chunk together in the original shape.
    peak_dirs_array = np.zeros(data_shape[0:3] + (npeaks, 3))
    peak_values_array = np.zeros(data_shape[0:3] + (npeaks,))
    peak_indices_array = np.zeros(data_shape[0:3] + (npeaks,))

    # tmp arrays are neccesary to avoid inserting data in returned variable
    # rather than the original array
    tmp_peak_dirs_array = np.zeros((np.count_nonzero(mask), npeaks, 3))
    tmp_peak_values_array = np.zeros((np.count_nonzero(mask), npeaks))
    tmp_peak_indices_array = np.zeros((np.count_nonzero(mask), npeaks))
    for i, peak_dirs, peak_values, peak_indices in results:
        tmp_peak_dirs_array[chunk_len[i]:chunk_len[i+1], :, :] = peak_dirs
        tmp_peak_values_array[chunk_len[i]:chunk_len[i+1], :] = peak_values
        tmp_peak_indices_array[chunk_len[i]:chunk_len[i+1], :] = peak_indices

    peak_dirs_array[mask] = tmp_peak_dirs_array
    peak_values_array[mask] = tmp_peak_values_array
    peak_indices_array[mask] = tmp_peak_indices_array

    return peak_dirs_array, peak_values_array, peak_indices_array


def _maps_from_sh_parallel(args):
    shm_coeff = args[0]
    _ = args[1]
    peak_values = args[2]
    peak_indices = args[3]
    B = args[4]
    sphere = args[5]
    gfa_thr = args[6]
    chunk_id = args[7]

    data_shape = shm_coeff.shape[0]
    nufo_map = np.zeros(data_shape)
    afd_max = np.zeros(data_shape)
    afd_sum = np.zeros(data_shape)
    rgb_map = np.zeros((data_shape, 3))
    gfa_map = np.zeros(data_shape)
    qa_map = np.zeros((data_shape, peak_values.shape[1]))

    max_odf = 0
    global_max = -np.inf
    for idx in range(len(shm_coeff)):
        if shm_coeff[idx].any():
            odf = np.dot(shm_coeff[idx], B)
            odf = odf.clip(min=0)
            sum_odf = np.sum(odf)
            max_odf = np.maximum(max_odf, sum_odf)
            if sum_odf > 0:
                rgb_map[idx] = np.dot(np.abs(sphere.vertices).T, odf)
                rgb_map[idx] /= np.linalg.norm(rgb_map[idx])
                rgb_map[idx] *= sum_odf
            gfa_map[idx] = gfa(odf)
            if gfa_map[idx] < gfa_thr:
                global_max = max(global_max, odf.max())
            elif np.sum(peak_indices[idx] > -1):
                nufo_map[idx] = np.sum(peak_indices[idx] > -1)
                afd_max[idx] = peak_values[idx].max()
                afd_sum[idx] = np.sqrt(np.dot(shm_coeff[idx], shm_coeff[idx]))
                qa_map = peak_values[idx] - odf.min()
                global_max = max(global_max, peak_values[idx][0])

    return chunk_id, nufo_map, afd_max, afd_sum, rgb_map, \
        gfa_map, qa_map, max_odf, global_max


def maps_from_sh(shm_coeff, peak_dirs, peak_values, peak_indices, sphere,
                 mask=None, gfa_thr=0, sh_basis_type='descoteaux07',
                 nbr_processes=None):
    """Computes maps from given SH coefficients and peaks

    Parameters
    ----------
    shm_coeff : np.ndarray
        Spherical harmonic coefficients
    peak_dirs : np.ndarray
        Peak directions
    peak_values : np.ndarray
        Peak values
    peak_indices : np.ndarray
        Peak indices
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    mask : np.ndarray, optional
        If `mask` is provided, only the data inside the mask will be
        used for computations.
    gfa_thr : float, optional
        Voxels with gfa less than `gfa_thr` are skipped for all metrics, except
        `rgb_map`.
        Default: 0
    sh_basis_type : str, optional
        Type of spherical harmonic basis used for `shm_coeff`. Either
        `descoteaux07` or `tournier07`.
        Default: `descoteaux07`
    nbr_processes: int, optional
        The number of subprocesses to use.
        Default: multiprocessing.cpu_count()

    Returns
    -------
    tuple of np.ndarray
        nufo_map, afd_max, afd_sum, rgb_map, gfa, qa
    """
    sh_order = order_from_ncoef(shm_coeff.shape[-1])
    B, _ = sh_to_sf_matrix(sphere, sh_order, sh_basis_type)

    data_shape = shm_coeff.shape
    if mask is None:
        mask = np.sum(shm_coeff, axis=3).astype(bool)

    nbr_processes = multiprocessing.cpu_count() \
        if nbr_processes is None or nbr_processes < 0 \
        else nbr_processes

    npeaks = peak_values.shape[3]
    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    shm_coeff = shm_coeff[mask].reshape(
        (np.count_nonzero(mask), data_shape[3]))
    peak_dirs = peak_dirs[mask].reshape((np.count_nonzero(mask), npeaks, 3))
    peak_values = peak_values[mask].reshape((np.count_nonzero(mask), npeaks))
    peak_indices = peak_indices[mask].reshape((np.count_nonzero(mask), npeaks))
    shm_coeff_chunks = np.array_split(shm_coeff, nbr_processes)
    peak_dirs_chunks = np.array_split(peak_dirs, nbr_processes)
    peak_values_chunks = np.array_split(peak_values, nbr_processes)
    peak_indices_chunks = np.array_split(peak_indices, nbr_processes)
    chunk_len = np.cumsum([0] + [len(c) for c in shm_coeff_chunks])

    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(_maps_from_sh_parallel,
                       zip(shm_coeff_chunks,
                           peak_dirs_chunks,
                           peak_values_chunks,
                           peak_indices_chunks,
                           itertools.repeat(B),
                           itertools.repeat(sphere),
                           itertools.repeat(gfa_thr),
                           np.arange(len(shm_coeff_chunks))))
    pool.close()
    pool.join()

    # Re-assemble the chunk together in the original shape.
    nufo_map_array = np.zeros(data_shape[0:3])
    afd_max_array = np.zeros(data_shape[0:3])
    afd_sum_array = np.zeros(data_shape[0:3])
    rgb_map_array = np.zeros(data_shape[0:3] + (3,))
    gfa_map_array = np.zeros(data_shape[0:3])
    qa_map_array = np.zeros(data_shape[0:3] + (npeaks,))

    # tmp arrays are neccesary to avoid inserting data in returned variable
    # rather than the original array
    tmp_nufo_map_array = np.zeros((np.count_nonzero(mask)))
    tmp_afd_max_array = np.zeros((np.count_nonzero(mask)))
    tmp_afd_sum_array = np.zeros((np.count_nonzero(mask)))
    tmp_rgb_map_array = np.zeros((np.count_nonzero(mask), 3))
    tmp_gfa_map_array = np.zeros((np.count_nonzero(mask)))
    tmp_qa_map_array = np.zeros((np.count_nonzero(mask), npeaks))

    all_time_max_odf = -np.inf
    all_time_global_max = -np.inf
    for (i, nufo_map, afd_max, afd_sum, rgb_map,
         gfa_map, qa_map, max_odf, global_max) in results:
        all_time_max_odf = max(all_time_global_max, max_odf)
        all_time_global_max = max(all_time_global_max, global_max)

        tmp_nufo_map_array[chunk_len[i]:chunk_len[i+1]] = nufo_map
        tmp_afd_max_array[chunk_len[i]:chunk_len[i+1]] = afd_max
        tmp_afd_sum_array[chunk_len[i]:chunk_len[i+1]] = afd_sum
        tmp_rgb_map_array[chunk_len[i]:chunk_len[i+1], :] = rgb_map
        tmp_gfa_map_array[chunk_len[i]:chunk_len[i+1]] = gfa_map
        tmp_qa_map_array[chunk_len[i]:chunk_len[i+1], :] = qa_map

    nufo_map_array[mask] = tmp_nufo_map_array
    afd_max_array[mask] = tmp_afd_max_array
    afd_sum_array[mask] = tmp_afd_sum_array
    rgb_map_array[mask] = tmp_rgb_map_array
    gfa_map_array[mask] = tmp_gfa_map_array
    qa_map_array[mask] = tmp_qa_map_array

    rgb_map_array /= all_time_max_odf
    rgb_map_array *= 255
    qa_map_array /= all_time_global_max

    afd_unique = np.unique(afd_max_array)
    if np.array_equal(np.array([0, 1]), afd_unique) \
            or np.array_equal(np.array([1]), afd_unique):
        logging.warning('All AFD_max values are 1. The peaks seem normalized.')

    return(nufo_map_array, afd_max_array, afd_sum_array,
           rgb_map_array, gfa_map_array, qa_map_array)


def _convert_sh_basis_parallel(args):
    sh = args[0]
    B_in = args[1]
    invB_out = args[2]
    chunk_id = args[3]

    for idx in range(sh.shape[0]):
        if sh[idx].any():
            sf = np.dot(sh[idx], B_in)
            sh[idx] = np.dot(sf, invB_out)

    return chunk_id, sh


def convert_sh_basis(shm_coeff, sphere, mask=None,
                     input_basis='descoteaux07', output_basis='tournier07',
                     is_input_legacy=True, is_output_legacy=False,
                     nbr_processes=None):
    """Converts spherical harmonic coefficients between two bases

    Parameters
    ----------
    shm_coeff : np.ndarray
        Spherical harmonic coefficients
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    mask : np.ndarray, optional
        If `mask` is provided, only the data inside the mask will be
        used for computations.
    input_basis : str, optional
        Type of spherical harmonic basis used for `shm_coeff`. Either
        `descoteaux07` or `tournier07`.
        Default: `descoteaux07`
    output_basis : str, optional
        Type of spherical harmonic basis wanted as output. Either
        `descoteaux07` or `tournier07`.
        Default: `tournier07`
    is_input_legacy: bool, optional
        If true, this means that the input SH used a legacy basis definition
        for backward compatibility with previous ``tournier07`` and
        ``descoteaux07`` implementations.
        Default: True
    is_output_legacy: bool, optional
        If true, this means that the output SH will use a legacy basis
        definition for backward compatibility with previous ``tournier07`` and
        ``descoteaux07`` implementations.
        Default: False
    nbr_processes: int, optional
        The number of subprocesses to use.
        Default: multiprocessing.cpu_count()

    Returns
    -------
    shm_coeff_array : np.ndarray
        Spherical harmonic coefficients in the desired basis.
    """
    if input_basis == output_basis and is_input_legacy == is_output_legacy:
        logging.info('Input and output SH basis are equal, no SH basis '
                     'convertion needed.')
        return shm_coeff

    sh_order = order_from_ncoef(shm_coeff.shape[-1])
    B_in, _ = sh_to_sf_matrix(sphere, sh_order, input_basis,
                              legacy=is_input_legacy)
    _, invB_out = sh_to_sf_matrix(sphere, sh_order, output_basis,
                                  legacy=is_output_legacy)

    data_shape = shm_coeff.shape
    if mask is None:
        mask = np.sum(shm_coeff, axis=3).astype(bool)

    nbr_processes = multiprocessing.cpu_count() \
        if nbr_processes is None or nbr_processes < 0 else nbr_processes

    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    shm_coeff = shm_coeff[mask].reshape(
        (np.count_nonzero(mask), data_shape[3]))
    shm_coeff_chunks = np.array_split(shm_coeff, nbr_processes)
    chunk_len = np.cumsum([0] + [len(c) for c in shm_coeff_chunks])

    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(_convert_sh_basis_parallel,
                       zip(shm_coeff_chunks,
                           itertools.repeat(B_in),
                           itertools.repeat(invB_out),
                           np.arange(len(shm_coeff_chunks))))
    pool.close()
    pool.join()

    # Re-assemble the chunk together in the original shape.
    shm_coeff_array = np.zeros(data_shape)
    tmp_shm_coeff_array = np.zeros((np.count_nonzero(mask), data_shape[3]))
    for i, new_shm_coeff in results:
        tmp_shm_coeff_array[chunk_len[i]:chunk_len[i+1], :] = new_shm_coeff

    shm_coeff_array[mask] = tmp_shm_coeff_array

    return shm_coeff_array


def _convert_sh_to_sf_parallel(args):
    sh = args[0]
    B_in = args[1]
    new_output_dim = args[2]
    chunk_id = args[3]
    sf = np.zeros((sh.shape[0], new_output_dim), dtype=np.float32)

    for idx in range(sh.shape[0]):
        if sh[idx].any():
            sf[idx] = np.dot(sh[idx], B_in)

    return chunk_id, sf


def convert_sh_to_sf(shm_coeff, sphere, mask=None, dtype="float32",
                     input_basis='descoteaux07', input_full_basis=False,
                     is_input_legacy=True,
                     nbr_processes=multiprocessing.cpu_count()):
    """Converts spherical harmonic coefficients to an SF sphere

    Parameters
    ----------
    shm_coeff : np.ndarray
        Spherical harmonic coefficients
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    mask : np.ndarray, optional
        If `mask` is provided, only the data inside the mask will be
        used for computations.
    dtype : str
        Datatype to use for computation and output array.
        Either `float32` or `float64`. Default: `float32`
    input_basis : str, optional
        Type of spherical harmonic basis used for `shm_coeff`. Either
        `descoteaux07` or `tournier07`.
        Default: `descoteaux07`
    input_full_basis : bool, optional
        If True, use a full SH basis (even and odd orders) for the input SH
        coefficients.
    is_input_legacy : bool, optional
        Whether the input basis is in its legacy form.
    nbr_processes: int, optional
        The number of subprocesses to use.
        Default: multiprocessing.cpu_count()

    Returns
    -------
    shm_coeff_array : np.ndarray
        Spherical harmonic coefficients in the desired basis.
    """
    assert dtype in ["float32", "float64"], "Only `float32` and `float64` " \
                                            "should be used."

    sh_order = order_from_ncoef(shm_coeff.shape[-1],
                                full_basis=input_full_basis)
    B_in, _ = sh_to_sf_matrix(sphere, sh_order, basis_type=input_basis,
                              full_basis=input_full_basis,
                              legacy=is_input_legacy)
    B_in = B_in.astype(dtype)

    data_shape = shm_coeff.shape
    if mask is None:
        mask = np.sum(shm_coeff, axis=3).astype(bool)

    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    shm_coeff = shm_coeff[mask].reshape(
        (np.count_nonzero(mask), data_shape[3]))
    shm_coeff_chunks = np.array_split(shm_coeff, nbr_processes)
    chunk_len = np.cumsum([0] + [len(c) for c in shm_coeff_chunks])

    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(_convert_sh_to_sf_parallel,
                       zip(shm_coeff_chunks,
                           itertools.repeat(B_in),
                           itertools.repeat(len(sphere.vertices)),
                           np.arange(len(shm_coeff_chunks))))
    pool.close()
    pool.join()

    # Re-assemble the chunk together in the original shape.
    new_shape = data_shape[:3] + (len(sphere.vertices),)
    sf_array = np.zeros(new_shape, dtype=dtype)
    tmp_sf_array = np.zeros((np.count_nonzero(mask), new_shape[3]),
                            dtype=dtype)
    for i, new_sf in results:
        tmp_sf_array[chunk_len[i]:chunk_len[i + 1], :] = new_sf

    sf_array[mask] = tmp_sf_array

    return sf_array
