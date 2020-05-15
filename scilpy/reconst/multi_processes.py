import itertools
import logging
import multiprocessing

from dipy.direction.peaks import peak_directions
from dipy.reconst.multi_voxel import MultiVoxelFit
from dipy.reconst.odf import gfa
from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
from dipy.segment.mask import applymask
import numpy as np


def fit_from_model_parallel(args):
    model = args[0]
    data = args[1]
    chunk_id = args[2]

    sub_fit_array = np.zeros((data.shape[0],), dtype='object')
    for i in range(data.shape[0]):
        if data[i].any():
            sub_fit_array[i] = model.fit(data[i])

    return chunk_id, sub_fit_array


def fit_from_model(model, data, mask=None, nbr_processes=None):
    """Fit the model to data

    Parameters
    ----------
    model : a model instance
        `model` will be used to fit the data.
    data : np.ndarray (4d)
        Diffusion data.
    mask : np.ndarray, optional
        If `mask` is provided, only the data inside the mask will be
        used for computations.
    nbr_processes : int, optional
        The number of subprocesses to use.
        Default: multiprocessing.cpu_count()

    Returns
    -------
    fit_array : np.ndarray
        Array containing the fit
    """
    data_shape = data.shape
    if mask is None:
        mask = np.sum(data, axis=3).astype(bool)
    else:
        data = applymask(data, mask)

    nbr_processes = multiprocessing.cpu_count() if nbr_processes is None \
        or nbr_processes <= 0 else nbr_processes

    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    data = data.ravel().reshape(np.prod(data_shape[0:3]), data_shape[3])
    chunks = np.array_split(data, nbr_processes)
    chunk_len = np.cumsum([0] + [len(c) for c in chunks])

    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(fit_from_model_parallel,
                       zip(itertools.repeat(model),
                           chunks,
                           np.arange(len(chunks))))
    pool.close()
    pool.join()

    # Re-assemble the chunk together in the original shape.
    fit_array = np.zeros((np.prod(data_shape[0:3]),), dtype='object')
    for i, fit in results:
        fit_array[chunk_len[i]:chunk_len[i+1]] = fit
    fit_array = MultiVoxelFit(model, fit_array.reshape(data_shape[0:3]), mask)

    return fit_array


def peaks_from_sh_parallel(args):
    shm_coeff = args[0]
    B = args[1]
    sphere = args[2]
    relative_peak_threshold = args[3]
    absolute_threshold = args[4]
    min_separation_angle = args[5]
    npeaks = args[6]
    normalize_peaks = args[7]
    chunk_id = args[8]

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
                                            min_separation_angle)

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
                  sh_basis_type='descoteaux07', nbr_processes=None):
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
        voxels (ex. ventricles). `scil_compute_fodf_max_in_ventricles.py`
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
    nbr_processes: int, optional
        The number of subprocesses to use.
        Default: multiprocessing.cpu_count()

    Returns
    -------
    tuple of np.ndarray
        peak_dirs, peak_values, peak_indices
    """
    sh_order = order_from_ncoef(shm_coeff.shape[-1])
    B, _ = sh_to_sf_matrix(sphere, sh_order, sh_basis_type)

    data_shape = shm_coeff.shape
    if mask is not None:
        shm_coeff = applymask(shm_coeff, mask)

    nbr_processes = multiprocessing.cpu_count() if nbr_processes is None \
        or nbr_processes < 0 else nbr_processes

    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    shm_coeff = shm_coeff.ravel().reshape(np.prod(data_shape[0:3]), data_shape[3])
    chunks = np.array_split(shm_coeff, nbr_processes)
    chunk_len = np.cumsum([0] + [len(c) for c in chunks])

    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(peaks_from_sh_parallel,
                       zip(chunks,
                           itertools.repeat(B),
                           itertools.repeat(sphere),
                           itertools.repeat(relative_peak_threshold),
                           itertools.repeat(absolute_threshold),
                           itertools.repeat(min_separation_angle),
                           itertools.repeat(npeaks),
                           itertools.repeat(normalize_peaks),
                           np.arange(len(chunks))))
    pool.close()
    pool.join()

    # Re-assemble the chunk together in the original shape.
    peak_dirs_array = np.zeros((np.prod(data_shape[0:3]), npeaks, 3))
    peak_values_array = np.zeros((np.prod(data_shape[0:3]), npeaks))
    peak_indices_array = np.zeros((np.prod(data_shape[0:3]), npeaks))
    for i, peak_dirs, peak_values, peak_indices in results:
        peak_dirs_array[chunk_len[i]:chunk_len[i+1], :] = peak_dirs
        peak_values_array[chunk_len[i]:chunk_len[i+1], :] = peak_values
        peak_indices_array[chunk_len[i]:chunk_len[i+1], :] = peak_indices
    peak_dirs_array = peak_dirs_array.reshape(data_shape[0:3]+(npeaks, 3))
    peak_values_array = peak_values_array.reshape(data_shape[0:3]+(npeaks,))
    peak_indices_array = peak_indices_array.reshape(data_shape[0:3]+(npeaks,))

    return peak_dirs_array, peak_values_array, peak_indices_array


def maps_from_sh_parallel(args):
    shm_coeff = args[0]
    peak_dirs = args[1]
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

    rgb_map /= max_odf
    rgb_map *= 255
    qa_map /= global_max

    return chunk_id, nufo_map, afd_max, afd_sum, rgb_map, gfa_map, qa_map


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
    if mask is not None:
        shm_coeff = applymask(shm_coeff, mask)
        peak_dirs = applymask(peak_dirs, mask)
        peak_values = applymask(peak_values, mask)
        peak_indices = applymask(peak_indices, mask)

    nbr_processes = multiprocessing.cpu_count() if nbr_processes is None \
        or nbr_processes < 0 else nbr_processes

    npeaks = peak_values.shape[3]
    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    shm_coeff = shm_coeff.ravel().reshape(np.prod(data_shape[0:3]), data_shape[3])
    peak_dirs = peak_dirs.ravel().reshape((np.prod(data_shape[0:3]), npeaks, 3))
    peak_values = peak_values.ravel().reshape(np.prod(data_shape[0:3]), npeaks)
    peak_indices = peak_indices.ravel().reshape(np.prod(data_shape[0:3]), npeaks)
    shm_coeff_chunks = np.array_split(shm_coeff, nbr_processes)
    peak_dirs_chunks = np.array_split(peak_dirs, nbr_processes)
    peak_values_chunks = np.array_split(peak_values, nbr_processes)
    peak_indices_chunks = np.array_split(peak_indices, nbr_processes)
    chunk_len = np.cumsum([0] + [len(c) for c in shm_coeff_chunks])

    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(maps_from_sh_parallel,
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
    nufo_map_array = np.zeros((np.prod(data_shape[0:3])))
    afd_max_array = np.zeros((np.prod(data_shape[0:3])))
    afd_sum_array = np.zeros((np.prod(data_shape[0:3])))
    rgb_map_array = np.zeros((np.prod(data_shape[0:3]), 3))
    gfa_map_array = np.zeros((np.prod(data_shape[0:3])))
    qa_map_array = np.zeros((np.prod(data_shape[0:3]), npeaks))
    for i, nufo_map, afd_max, afd_sum, rgb_map, gfa_map, qa_map in results:
        nufo_map_array[chunk_len[i]:chunk_len[i+1]] = nufo_map
        afd_max_array[chunk_len[i]:chunk_len[i+1]] = afd_max
        afd_sum_array[chunk_len[i]:chunk_len[i+1]] = afd_sum
        rgb_map_array[chunk_len[i]:chunk_len[i+1], :] = rgb_map
        gfa_map_array[chunk_len[i]:chunk_len[i+1]] = gfa_map
        qa_map_array[chunk_len[i]:chunk_len[i+1], :] = qa_map
    nufo_map_array = nufo_map_array.reshape(data_shape[0:3])
    afd_max_array = afd_max_array.reshape(data_shape[0:3])
    afd_sum_array = afd_sum_array.reshape(data_shape[0:3])
    rgb_map_array = rgb_map_array.reshape(data_shape[0:3] + (3,))
    gfa_map_array = gfa_map_array.reshape(data_shape[0:3])
    qa_map_array = qa_map_array.reshape(data_shape[0:3] + (npeaks,))

    afd_unique = np.unique(afd_max_array)
    if np.array_equal(np.array([0, 1]), afd_unique) \
        or np.array_equal(np.array([1]), afd_unique):
        logging.warning('All AFD_max values are 1. The peaks seem normalized.')

    return(nufo_map_array, afd_max_array, afd_sum_array,
           rgb_map_array, gfa_map_array, qa_map_array)


def convert_sh_basis_parallel(args):
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
                     input_basis='descoteaux07', nbr_processes=None):
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
    nbr_processes: int, optional
        The number of subprocesses to use.
        Default: multiprocessing.cpu_count()

    Returns
    -------
    shm_coeff_array : np.ndarray
        Spherical harmonic coefficients in the desired basis.
    """
    output_basis = 'descoteaux07' if input_basis == 'tournier07' else 'tournier07'

    sh_order = order_from_ncoef(shm_coeff.shape[-1])
    B_in, _ = sh_to_sf_matrix(sphere, sh_order, input_basis)
    _, invB_out = sh_to_sf_matrix(sphere, sh_order, output_basis)

    data_shape = shm_coeff.shape
    if mask is not None:
        shm_coeff = applymask(shm_coeff, mask)

    nbr_processes = multiprocessing.cpu_count() if nbr_processes is None \
        or nbr_processes < 0 else nbr_processes

    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    shm_coeff = shm_coeff.ravel().reshape(np.prod(data_shape[0:3]), data_shape[3])
    shm_coeff_chunks = np.array_split(shm_coeff, nbr_processes)
    chunk_len = np.cumsum([0] + [len(c) for c in shm_coeff_chunks])

    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(convert_sh_basis_parallel,
                       zip(shm_coeff_chunks,
                           itertools.repeat(B_in),
                           itertools.repeat(invB_out),
                           np.arange(len(shm_coeff_chunks))))
    pool.close()
    pool.join()

    # Re-assemble the chunk together in the original shape.
    shm_coeff_array = np.zeros((np.prod(data_shape[0:3]), data_shape[3]))
    for i, new_shm_coeff in results:
        shm_coeff_array[chunk_len[i]:chunk_len[i+1], :] = new_shm_coeff
    shm_coeff_array = shm_coeff_array.reshape(data_shape[0:3]+(data_shape[3],))

    return shm_coeff_array
