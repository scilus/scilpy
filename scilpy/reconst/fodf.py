# -*- coding: utf-8 -*-
import itertools
import logging
import multiprocessing
import numpy as np

from dipy.data import get_sphere
from dipy.reconst.mcsd import MSDeconvFit
from dipy.reconst.multi_voxel import MultiVoxelFit
from dipy.reconst.shm import sh_to_sf_matrix

from scilpy.reconst.utils import find_order_from_nb_coeff

from dipy.utils.optpkg import optional_package
cvx, have_cvxpy, _ = optional_package("cvxpy")


def get_ventricles_max_fodf(data, fa, md, zoom, sh_basis,
                            fa_threshold, md_threshold,
                            small_dims=False, is_legacy=True):
    """
    Compute mean maximal fodf value in ventricules. Given heuristics thresholds
    on FA and MD values, finds the voxels of the ventricules or CSF and
    computes a mean fODF value. This is described in
    Dell'Acqua et al. HBM 2013.

    Ventricles are searched in a window in the middle of the data to increase
    speed. No need to scan the whole image.

    Parameters
    ----------
    data: ndarray (x, y, z, ncoeffs)
         Input fODF file in spherical harmonics coefficients. Uses sphere
         'repulsion100' to convert to SF values.
    fa: ndarray (x, y, z)
         FA (Fractional Anisotropy) volume from DTI
    md: ndarray (x, y, z)
         MD (Mean Diffusivity) volume from DTI
    zoom: List of length 3
        The resolution. A total number of voxels of 1000 works well at
        2x2x2 = 8 mm3.
    sh_basis: str
        Either 'tournier07' or 'descoteaux07'
    small_dims: bool
        If set, takes the full range of data to search the max fodf amplitude
        in ventricles, rather than a window center in the data. Useful when the
        data has small dimensions.
    fa_threshold: float
        Maximal threshold of FA (voxels under that threshold are considered
        for evaluation). Suggested value: 0.1.
    md_threshold: float
        Minimal threshold of MD in mm2/s (voxels above that threshold are
        considered for evaluation). Suggested value: 0.003.
    is_legacy : bool, optional
        Whether the SH basis is in its legacy form.

    Returns
    -------
    mean, mask: int, ndarray (x, y, z)
         Mean maximum fODF value and mask of voxels used.
    """

    order = find_order_from_nb_coeff(data)
    sphere = get_sphere('repulsion100')
    b_matrix, _ = sh_to_sf_matrix(sphere, order, sh_basis, legacy=is_legacy)
    mask = np.zeros(data.shape[:-1])

    # 1000 works well at 2x2x2 = 8 mm3
    # Hence, we multiply by the volume of a voxel
    vol = (zoom[0] * zoom[1] * zoom[2])
    if vol != 0:
        max_number_of_voxels = 1000 * 8 // vol
    else:
        max_number_of_voxels = 1000
    logging.debug("Searching for ventricle voxels, up to a maximum of {} "
                  "voxels.".format(max_number_of_voxels))

    # In the case of 2D-like data (3D data with one dimension size of 1), or
    # a small 3D dataset, the full range of data is scanned.
    if small_dims:
        all_i = list(range(0, data.shape[0]))
        all_j = list(range(0, data.shape[1]))
        all_k = list(range(0, data.shape[2]))
    # In the case of a normal 3D dataset, a window is created in the middle of
    # the image to capture the ventricles. No need to scan the whole image.
    # (Automatic definition of window's radius based on the shape of the data.)
    else:
        if np.min(data.shape[:-1]) > 40:
            radius = 20
        else:
            if np.min(data.shape[:-1]) > 20:
                radius = 10
            else:
                radius = 5

        all_i = list(range(int(data.shape[0]/2) - radius,
                           int(data.shape[0]/2) + radius))
        all_j = list(range(int(data.shape[1]/2) - radius,
                           int(data.shape[1]/2) + radius))
        all_k = list(range(int(data.shape[2]/2) - radius,
                           int(data.shape[2]/2) + radius))

    # Ok. Now find ventricle voxels.
    sum_of_max = 0
    count = 0
    for i in all_i:
        for j in all_j:
            for k in all_k:
                if count > max_number_of_voxels - 1:
                    continue
                if fa[i, j, k] < fa_threshold \
                        and md[i, j, k] > md_threshold:
                    sf = np.dot(data[i, j, k], b_matrix)
                    sum_of_max += sf.max()
                    count += 1
                    mask[i, j, k] = 1

    logging.info('Number of voxels detected: {}'.format(count))
    if count == 0:
        logging.warning('No voxels found for evaluation! Change your fa '
                        'and/or md thresholds')
        return 0, mask

    logging.info('Average max fodf value: {}'.format(sum_of_max / count))
    return sum_of_max / count, mask


def _fit_from_model_parallel(args):
    model = args[0]
    data = args[1]
    chunk_id = args[2]

    sub_fit_array = np.zeros((data.shape[0],), dtype='object')
    for i in range(data.shape[0]):
        if data[i].any():
            try:
                sub_fit_array[i] = model.fit(data[i])
            except cvx.error.SolverError:
                coeff = np.full((len(model.n)), np.NaN)
                sub_fit_array[i] = MSDeconvFit(model, coeff, None)

    return chunk_id, sub_fit_array


def fit_from_model(model, data, mask=None, nbr_processes=None):
    """Fit the model to data. Can use parallel processing.

    Parameters
    ----------
    model : a model instance
        It will be used to fit the data.
        e.g: An instance of dipy.reconst.shm.SphHarmFit.
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
    fit_array : MultiVoxelFit
        Dipy's MultiVoxelFit, containing the fit.
        It contains an array of fits. Any attributes of its individuals fits
        (of class given by 'model.fit') can be accessed through the
        MultiVoxelFit to get all fits at once.
    """
    data_shape = data.shape
    if mask is None:
        mask = np.sum(data, axis=3).astype(bool)
    else:
        mask_any = np.sum(data, axis=3).astype(bool)
        mask *= mask_any

    nbr_processes = multiprocessing.cpu_count() \
        if nbr_processes is None or nbr_processes <= 0 \
        else nbr_processes

    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    data = data[mask].reshape((np.count_nonzero(mask), data_shape[3]))
    chunks = np.array_split(data, nbr_processes)

    chunk_len = np.cumsum([0] + [len(c) for c in chunks])
    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(_fit_from_model_parallel,
                       zip(itertools.repeat(model),
                           chunks,
                           np.arange(len(chunks))))
    pool.close()
    pool.join()

    # Re-assemble the chunk together in the original shape.
    fit_array = np.zeros(data_shape[0:3], dtype='object')
    tmp_fit_array = np.zeros((np.count_nonzero(mask)), dtype='object')
    for i, fit in results:
        tmp_fit_array[chunk_len[i]:chunk_len[i+1]] = fit

    fit_array[mask] = tmp_fit_array
    fit_array = MultiVoxelFit(model, fit_array, mask)

    return fit_array


def verify_failed_voxels_shm_coeff(shm_coeff):
    """
    Verifies if there are any NaN in the final coefficients, and if so raises
    warnings.

    Parameters
    ----------
    shm_coeff: np.ndarray
        The shm_coeff given by dipy's fit classes. Of shape (x, y, z, n) with
        the coefficients on the last dimension.

    Returns
    -------
    shm_coeff: np.ndarray
        The coefficients with 0 instead of NaNs.
    """
    nan_count = len(np.argwhere(np.isnan(shm_coeff[..., 0])))
    voxel_count = np.prod(shm_coeff.shape[:-1])

    if nan_count / voxel_count >= 0.05:
        logging.warning(
            "There are {} voxels out of {} that could not be solved by the "
            "solver, reaching a critical amount of voxels. Make sure to tune "
            "the response functions properly, as the solving process is very "
            "sensitive to it. Proceeding to fill the problematic voxels by 0."
            .format(nan_count, voxel_count))
    elif nan_count > 0:
        logging.warning(
            "There are {} voxels out of {} that could not be solved by the "
            "solver. Make sure to tune the response functions properly, as "
            "the solving process is very sensitive to it. Proceeding to fill "
            "the problematic voxels by 0.".format(nan_count, voxel_count))

    shm_coeff = np.where(np.isnan(shm_coeff), 0, shm_coeff)
    return shm_coeff


def verify_frf_files(wm_frf, gm_frf, csf_frf):
    """
    Verifies that all three frf files contain four columns, else raises
    ValueErrors.

    Parameters
    ----------
    wm_frf: np.ndarray
        The frf directly as loaded from its text file.
    gm_frf: np.ndarray
        Idem
    csf_frf: np.ndarray
        Idem

    Returns
    -------
    wm_frf: np.ndarray
        The file. In the case where there was only one line in the file, and it
        has been loaded as a vector, we return the array formatted as 2D, with
        the 4 frf values as columns.
    gm_frf: np.ndarray
        Idem
    csf_frf: np.ndarray
        Idem
    """
    if len(wm_frf.shape) == 1:
        wm_frf = wm_frf[None, :]
    if len(gm_frf.shape) == 1:
        gm_frf = gm_frf[None, :]
    if len(csf_frf.shape) == 1:
        csf_frf = csf_frf[None, :]

    if not wm_frf.shape[1] == 4:
        raise ValueError('WM frf file did not contain 4 elements. '
                         'Invalid or deprecated FRF format')
    if not gm_frf.shape[1] == 4:
        raise ValueError('GM frf file did not contain 4 elements. '
                         'Invalid or deprecated FRF format')
    if not csf_frf.shape[1] == 4:
        raise ValueError('CSF frf file did not contain 4 elements. '
                         'Invalid or deprecated FRF format')

    return wm_frf, gm_frf, csf_frf
