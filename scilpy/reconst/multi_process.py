import itertools
import multiprocessing
import sys
​
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, default_sphere
from dipy.direction.peaks import peak_directions
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import auto_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.multi_voxel import MultiVoxelFit
from dipy.reconst.odf import gfa
from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
​
import numpy as np
​
​
def fit_from_model_parallel(args):
    model = args[0]
    data = args[1]
    chunk_id = args[2]
​
    sub_fit_array = np.zeros((data.shape[0],), dtype='object')
    for i in range(data.shape[0]):
        if data[i].any():
            sub_fit_array[i] = model.fit(data[i])
​
    return chunk_id, sub_fit_array
​
​
def fit_from_model(model, data, mask=None,
                   nbr_processes=None):
    """Fit the model to data
​
    Parameters
    ----------
    model : a model instance
        `model` will be used to fit the data.
    data : np.ndarray (4d)
        diffusion data.
    mask : array, optional
        If `mask` is provided, voxels that are False in `mask` are skipped and
        no peaks are returned.
    nbr_processes: int
        The number of subprocesses to use. 
        Default: cpu_count().
​
    Returns
    -------
    fit_array : np.ndarray
        Array containing the fit
    """
    data_shape = data.shape
    if mask is None:
        mask = np.sum(data, axis=3).astype(np.int32)
        data[mask == 0] = 0
    else:
        # Check shape before in main()
        data[mask == 0, :] = 0
​
    nbr_processes = multiprocessing.cpu_count() if nbr_processes is None \
        or nbr_processes <= 0 else nbr_processes
​
    data = data.ravel().reshape(np.prod(data_shape[0:3]), data_shape[3])
    chunks = np.array_split(data, nbr_processes)
    chunk_len = np.cumsum([0] + [len(c) for c in chunks])
​
    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(fit_from_model_parallel,
                       zip(itertools.repeat(model),
                           chunks,
                           np.arange(len(chunks))))
    pool.close()
    pool.join()
​
    fit_array = np.zeros((np.prod(data_shape[0:3]),), dtype='object')
    for i, fit in results:
        fit_array[chunk_len[i]:chunk_len[i+1]] = fit
​
    fit_array = MultiVoxelFit(model, fit_array.reshape(data_shape[0:3]), mask)
    return fit_array
​
​
def peaks_from_sh_parallel(args):
    shm_coeff = args[0]
    B = args[1]
    sphere = args[2]
    relative_peak_threshold = args[3]
    min_separation_angle = args[4]
    npeaks = args[5]
    normalize_peaks = args[6]
    chunk_id = args[7]
​
    data_shape = shm_coeff.shape[0]
    peak_dirs = np.zeros((data_shape, npeaks, 3))
    peak_values = np.zeros((data_shape, npeaks))
    peak_indices = np.zeros((data_shape, npeaks), dtype='int')
    peak_indices.fill(-1)
​
    for idx in range(len(shm_coeff)):
        odf = np.dot(shm_coeff[idx], B)
        dirs, peaks, ind = peak_directions(odf, sphere,
                                             relative_peak_threshold,
                                             min_separation_angle)
        # Calculate peak metrics
        if peaks.shape[0] != 0:
            n = min(npeaks, peaks.shape[0])
​
            peak_dirs[idx][:n] = dirs[:n]
            peak_indices[idx][:n] = ind[:n]
            peak_values[idx][:n] = peaks[:n]
​
            if normalize_peaks:
                peak_values[idx][:n] /= peaks[0]
                peak_dirs[idx] *= peak_values[idx][:, None]
​
    return chunk_id, peak_dirs, peak_values, peak_indices
​
​
def peaks_from_sh(shm_coeff, sphere, mask=None, relative_peak_threshold=0.5,
                   min_separation_angle=25, normalize_peaks=False,
                   npeaks=5, sh_basis_type='descoteaux07',
                   nbr_processes=None):
    """Fit the model to data and computes peaks and metrics
​
    Parameters
    ----------
    shm_coeff : np.ndarray
        Spherical harmonic coefficients
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    mask : array, optional
        If `mask` is provided, voxels that are False in `mask` are skipped and
        no peaks are returned.
    relative_peak_threshold : float, optional
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90], optional
        The minimum distance between
        directions. If two peaks are too close only the larger of the two is
        returned.
    normalize_peaks : bool, optional
        If true, all peak values are calculated relative to `max(odf)`.
    npeaks : int, optional
        Maximum number of peaks found (default 5 peaks).
    nbr_processes: int
        If `parallel` is True, the number of subprocesses to use
        (default multiprocessing.cpu_count()).
​
    Returns
    -------
    tuple of np.ndarray
        peak_dirs, peak_values, peak_indices
    """
    B, _ = sh_to_sf_matrix(sphere, order_from_ncoef(shm_coeff.shape[-1]), sh_basis_type)
​
    # I hate doing that kind of testing in a core function, this should be handle by the lawer above it to 'clarify' the code
    data_shape = shm_coeff.shape
    if mask is None:
        mask = np.sum(data, axis=3).astype(np.int32)
        data[mask == 0] = 0
    else:
        # Check shape before in main()
        data[mask == 0, :] = 0
​
    nbr_processes = multiprocessing.cpu_count() if nbr_processes is None \
        or nbr_processes < 0 else nbr_processes
​
    shm_coeff = shm_coeff.ravel().reshape(np.prod(data_shape[0:3]), data_shape[3])
    chunks = np.array_split(shm_coeff, nbr_processes)
    chunk_len = np.cumsum([0] + [len(c) for c in chunks])
​
    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(peaks_from_sh_parallel,
                       zip(chunks,
                           itertools.repeat(B),
                           itertools.repeat(sphere),
                           itertools.repeat(relative_peak_threshold),
                           itertools.repeat(min_separation_angle),
                           itertools.repeat(npeaks),
                           itertools.repeat(normalize_peaks),
                           np.arange(len(chunks))))
    pool.close()
    pool.join()
​
    peak_dirs_array = np.zeros((np.prod(data_shape[0:3]), npeaks, 3))
    peak_values_array = np.zeros((np.prod(data_shape[0:3]), npeaks))
    peak_indices_array = np.zeros((np.prod(data_shape[0:3]), npeaks))
    for i, peak_dirs, peak_values, peak_indices in results:
        peak_dirs_array[chunk_len[i]:chunk_len[i+1], :] = peak_dirs
        peak_values_array[chunk_len[i]:chunk_len[i+1], :] = peak_values
        peak_indices_array[chunk_len[i]:chunk_len[i+1], :] = peak_indices
​
    peak_dirs_array = peak_dirs_array.reshape(data_shape[0:3]+(npeaks, 3))
    peak_values_array = peak_values_array.reshape(data_shape[0:3]+(npeaks,))
    peak_indices_array = peak_indices_array.reshape(data_shape[0:3]+(npeaks,))
​
    return peak_dirs_array, peak_values_array, peak_indices_array


def maps_from_sh_parallel(args):
    shm_coeff = args[0]
    peaks_dirs = args[1]
    peaks_values = args[2]
    B = args[3]
    sphere = args[4]
    gfa_thr = args[5]
    chunk_id = args[6]
​
    data_shape = shm_coeff.shape[0]
    nufo_map = np.zeros(data_shape)
    afd_map = np.zeros(data_shape)
    afd_sum = np.zeros(data_shape)
    rgb_map = np.zeros(data_shape + (3,))

    gfa = np.zeros(data_shape)
    qa = np.zeros(data_shape + (peaks_values.shape[1],))
​
    max_odf = 0
    global_max = -np.inf
    for idx in range(len(shm_coeff)):
        if not np.isnan(data[idx]).any():
            odf = np.dot(shm_coeff[idx], B)
            sum_odf = np.sum(odf)
            max_odf = np.maximum(max_odf, sum_odf)
            if sum_odf > 0:
                rgb_map[ind] = np.sum(np.abs(sphere.vertices) * odf, axis=0)
                rgb_map[ind] /= np.linalg.norm(rgb_map[ind])
                rgb_map[ind] *= sum_odf
            gfa[idx] = gfa(odf)
            if gfa[idx] < gfa_thr:
                global_max = max(global_max, odf.max())
            elif len(np.linalg.norm(peaks_dir[idx], axis=1) > 0):
                nufo_map[idx] = peaks_dirs[idx].shape[0]
                afd_map[idx] = peaks_values[idx].max()
                afd_sum[idx] = n.sqrt(np.dot(shm_coeff[idx], shm_coeff[idx]))
                qa = peaks_values[idx] - odf.min()
                global_max = max(global_max, peaks_values[idx][0])

    rgb_map /= max_odf
    rgb_map *= 255
    qa /= global_max
​
    return chunk_id, nufo_map, afd_map, afd_sum, rgb_map, gfa, qa


def maps_from_sh(shm_coeff, peaks_dirs, peaks_values, sphere, mask=None,
                 sh_basis_type='descoteaux07', gfa_thr=0,
                 nbr_processes=None):
    B, _ = sh_to_sf_matrix(sphere, order_from_ncoef(shm_coeff.shape[-1]), sh_basis_type)

    # I hate doing that kind of testing in a core function, this should be handle by the lawer above it to 'clarify' the code
    data_shape = shm_coeff.shape
    if mask is None:
        mask = np.sum(data, axis=3).astype(np.int32)
        data[mask == 0] = 0
    else:
        # Check shape before in main()
        data[mask == 0, :] = 0
​
    nbr_processes = multiprocessing.cpu_count() if nbr_processes is None \
        or nbr_processes < 0 else nbr_processes

    # In the script : need to make sure that shm and peaks have the same data_shape
    shm_coeff = shm_coeff.ravel().reshape(np.prod(data_shape[0:3]), data_shape[3])
    peaks_dirs = peaks_dirs.ravel().reshape(np.prod(data_shape[0:3]), peaks_dirs.shape[3:])
    peaks_values = peaks_values.ravel().reshape(np.prod(data_shape[0:3]), peaks_values.shape[3])
    shm_coeff_chunks = np.array_split(shm_coeff, nbr_processes)
    peaks_dirs_chunks = np.array_split(peaks_dirs, nbr_processes)
    peaks_values_chunks = np.array_split(peaks_values, nbr_processes)
    chunk_len = np.cumsum([0] + [len(c) for c in shm_coeff_chunks])

    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(maps_from_sh_parallel,
                       zip(shm_coeff_chunks,
                           peaks_dirs_chunks,
                           peaks_values_chunks,
                           intertools.repeat(B),
                           itertools.repeat(sphere),
                           itertools.repeat(gfa_thr),
                           np.arange(len(shm_coeff_chunks))))
    pool.close()
    pool.join()

    nufo_map_array = np.zeros((np.prod(data_shape[0:3])))
    afd_map_array = np.zeros((np.prod(data_shape[0:3])))
    afd_sum_array = np.zeros((np.prod(data_shape[0:3])))
    rgb_map_array = np.zeros((np.prod(data_shape[0:3]), 3))
    gfa_array = np.zeros((np.prod(data_shape[0:3])))
    qa_array = np.zeros((np.prod(data_shape[0:3]), peaks_values.shape[3]))
    for i, nufo_map, afd_map, afd_sum, rgb_map, gfa, qa in results:
        nufo_map_array[chunk_len[i]:chunk_len[i+1]] = nufo_map
        afd_map_array[chunk_len[i]:chunk_len[i+1]] = afd_map
        afd_sum_array[chunk_len[i]:chunk_len[i+1]] = afd_sum
        rgb_map_array[chunk_len[i]:chunk_len[i+1], :] = rgb_map
        gfa_array[chunk_len[i]:chunk_len[i+1]] = gfa
        qa_array[chunk_len[i]:chunk_len[i+1], :] = qa
​
    nufo_map_array = nufo_map_array.reshape(data_shape[0:3])
    afd_map_array = afd_map_array.reshape(data_shape[0:3])
    afd_sum_array = afd_sum_array.reshape(data_shape[0:3])
    rgb_map_array = rgb_map_array.reshape(data_shape[0:3] + (3,))
    gfa_array = gfa_array.reshape(data_shape[0:3])
    qa_array = qa_array.reshape(data_shape[0:3] + (peaks_values.shape[3],))
​
    return(nufo_map_array, afd_map_array, afd_sum_array,
           rgb_map_array, gfa_array, qa_array)

def get_maps(data, mask, args, npeaks=5):
    nufo_map = np.zeros(data.shape[0:3])
    afd_map = np.zeros(data.shape[0:3])
    afd_sum = np.zeros(data.shape[0:3])

    peaks_dirs = np.zeros(list(data.shape[0:3]) + [npeaks, 3])
    order = find_order_from_nb_coeff(data)
    sphere = get_sphere(args.sphere)
    b_matrix = get_b_matrix(order, sphere, args.sh_basis)

    for index in ndindex(data.shape[:-1]):
        if mask[index]:
            if np.isnan(data[index]).any():
                nufo_map[index] = 0
                afd_map[index] = 0
            else:
                maximas, afd, _ = get_maximas(
                    data[index], sphere, b_matrix, args.r_threshold, args.at)
                # sf = np.dot(data[index], B.T)

                n = min(npeaks, maximas.shape[0])
                nufo_map[index] = maximas.shape[0]
                if n == 0:
                    afd_map[index] = 0.0
                    nufo_map[index] = 0.0
                else:
                    afd_map[index] = afd.max()
                    peaks_dirs[index][:n] = maximas[:n]

                    # sum of all coefficients, sqrt(power spectrum)
                    # sum C^2 = sum fODF^2
                    afd_sum[index] = np.sqrt(np.dot(data[index], data[index]))

                    # sum of all peaks contributions to the afd
                    # integral of all the lobes. Numerical sum.
                    # With an infinite number of SH, this should == to afd_sum
                    # sf[np.nonzero(sf < args.at)] = 0.
                    # afd_sum[index] = sf.sum()/n*4*np.pi/B.shape[0]x

    return nufo_map, afd_map, afd_sum, peaks_dirs
​