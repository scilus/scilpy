# -*- coding: utf-8 -*-
import itertools
import multiprocessing
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


def _get_bounds():
    """Define the lower (lb) and upper (ub) boundaries of the fitting
    parameters, being the signal without diffusion weighting (S0), the mean
    diffusivity (MD), the isotropic variance (V_I) and the anisotropic variance
    (V_A).

    Returns
    -------
    lb : list of floats
        Lower boundaries of the fitting parameters.
    ub : list of floats
        Upper boundaries of the fitting parameters.
    """
    S0 = [0, 10]
    MD = [1e-12, 4e-9]
    V_I = [1e-24, 5e-18]
    V_A = [1e-24, 5e-18]
    lb = [S0[0], MD[0], V_I[0], V_A[0]]
    ub = [S0[1], MD[1], V_I[1], V_A[1]]
    return lb, ub


def _random_p0(signal, gtab_infos, lb, ub, weight, n_iter):
    """Produce a guess of initial parameters for the fit, by calculating the
    signals of a given number of random sets of parameters and keeping the one
    closest to the input signal.

    Parameters
    ----------
    signal : np.ndarray
        Diffusion data of a single voxel.
    gtab_infos : np.ndarray
        Contains information about the gtab, such as the unique bvals, the
        encoding types, the number of directions and the acquisition index.
        Obtained as output of the function
        `io.btensor.generate_btensor_input`.
    lb : np.ndarray of floats
        Lower boundaries of the fitting parameters.
    ub : np.ndarray of floats
        Upper boundaries of the fitting parameters.
    weight : np.ndarray
        Gives a different weight to each element of `signal`.
    n_iter : int
        Number of random sets of parameters tested.

    Returns
    -------
    guess : np.ndarray
        Array containing the guessed initial parameters.
    """
    guess = []
    thr = np.inf

    for i in range(n_iter):
        params_rand = lb + (ub - lb) * np.random.rand(len(lb))
        signal_rand = _gamma_fit2data(gtab_infos, params_rand)
        residual_rand = np.sum(((signal - signal_rand) * weight)**2)

        if residual_rand < thr:
            thr = residual_rand
            guess = params_rand

    return guess


def _gamma_data2fit(signal, gtab_infos, fit_iters=1, random_iters=50,
                    do_weight_bvals=False, do_weight_pa=False,
                    do_multiple_s0=False):
    """Fit the gamma model to data

    Parameters
    ----------
    signal : np.array
        Diffusion data of a single voxel.
    gtab_infos : np.ndarray
        Contains information about the gtab, such as the unique bvals, the
        encoding types, the number of directions and the acquisition index.
        Obtained as output of the function
        `io.btensor.generate_btensor_input`.
    fit_iters : int, optional
        Number of iterations in the gamma fit. Defaults to 1.
    random_iters : int, optional
        Number of random sets of parameters tested to find the initial
        parameters. Defaults to 50.
    do_weight_bvals : bool , optional
        If set, does a weighting on the bvalues in the gamma fit.
    do_weight_pa : bool, optional
        If set, does a powder averaging weighting in the gamma fit.
    do_multiple_s0 : bool, optional
        If set, takes into account multiple baseline signals.

    Returns
    -------
    best_params : np.array
        Array containing the parameters of the fit.
    """
    if np.sum(gtab_infos[3]) > 0 and do_multiple_s0 is True:
        ns = len(np.unique(gtab_infos[3])) - 1
    else:
        ns = 0

    unit_to_SI = np.array([np.max(signal), 1e-9, 1e-18, 1e-18])
    unit_to_SI = np.concatenate((unit_to_SI, np.ones(ns)))

    def weight_bvals(sthr, mdthr, wthr):
        """Compute an array weighting the different components of the signal
        array based on the bvalue.
        """
        bthr = -np.log(sthr) / mdthr
        weight = 0.5 * (1 - erf(wthr * (gtab_infos[0] - bthr) / bthr))
        return weight

    def weight_pa():
        """Compute an array weighting the different components of the signal
        array based on the number of directions.
        """
        weight = np.sqrt(gtab_infos[2] / np.max(gtab_infos[2]))
        return weight

    def my_gamma_fit2data(gtab_infos, *args):
        """Compute a signal from gtab infomations and fit parameters.
        """
        params_unit = args
        params_SI = params_unit * unit_to_SI
        signal = _gamma_fit2data(gtab_infos, params_SI)
        return signal * weight

    lb_SI, ub_SI = _get_bounds()
    lb_SI = np.concatenate((lb_SI, 0.5 * np.ones(ns)))
    ub_SI = np.concatenate((ub_SI, 2.0 * np.ones(ns)))
    lb_SI[0] *= np.max(signal)
    ub_SI[0] *= np.max(signal)

    lb_unit = lb_SI / unit_to_SI
    ub_unit = ub_SI / unit_to_SI

    bounds_unit = ([lb_unit, ub_unit])

    res_thr = np.inf

    for i in range(fit_iters):
        weight = np.ones(len(signal))
        if do_weight_bvals:
            weight *= weight_bvals(0.07, 1e-9, 2)
        if do_weight_pa:
            weight *= weight_pa()

        p0_SI = _random_p0(signal, gtab_infos, lb_SI, ub_SI, weight,
                           random_iters)
        p0_unit = p0_SI / unit_to_SI
        params_unit, params_cov = curve_fit(my_gamma_fit2data, gtab_infos,
                                            signal * weight, p0=p0_unit,
                                            bounds=bounds_unit, method="trf",
                                            ftol=1e-8, xtol=1e-8, gtol=1e-8)

        if do_weight_bvals:
            weight = weight_bvals(0.07, params_unit[1] * unit_to_SI[1], 2)
            if do_weight_pa:
                weight *= weight_pa()

            params_unit, params_cov = curve_fit(my_gamma_fit2data, gtab_infos,
                                                signal * weight,
                                                p0=params_unit,
                                                bounds=bounds_unit,
                                                method="trf")

        signal_fit = _gamma_fit2data(gtab_infos, params_unit * unit_to_SI)
        residual = np.sum(((signal - signal_fit) * weight) ** 2)
        if residual < res_thr:
            res_thr = residual
            params_best = params_unit
            # params_cov_best = params_cov

    params_best[0] = params_best[0] * unit_to_SI[0]
    return params_best[0:4]


def _gamma_fit2data(gtab_infos, params):
    """Compute a signal from gtab infomations and fit parameters.

    Parameters
    ----------
    gtab_infos : np.ndarray
        Contains information about the gtab, such as the unique bvals, the
        encoding types, the number of directions and the acquisition index.
        Obtained as output of the function
        `io.btensor.generate_btensor_input`.
    params : np.array
        Array containing the parameters of the fit.

    Returns
    -------
    signal : np.array
        Array containing the signal produced by the gamma model.
    """
    S0 = params[0]
    MD = params[1]
    V_I = params[2]
    V_A = params[3]
    RS = params[4:]  # relative signal
    if len(RS) != 0:
        RS = np.concatenate(([1], RS))
        RS_tile = np.tile(RS, len(gtab_infos[0])).reshape((len(gtab_infos[0]),
                                                           len(RS)))
        RS_index = np.zeros((len(gtab_infos[0]), len(RS)))
        for i in range(len(gtab_infos[0])):
            j = gtab_infos[3][i]
            RS_index[i][int(j)] = 1
        RS_matrix = RS_tile * RS_index
        SW = S0 * np.sum(RS_matrix, axis=1)
    else:
        SW = S0

    V_D = V_I + V_A * (gtab_infos[1] ** 2)
    signal = SW * ((1 + gtab_infos[0] * V_D / MD) ** (-(MD ** 2) / V_D))

    return np.real(signal)


def gamma_fit2metrics(params):
    """Compute metrics from fit parameters. This is the only function that
    takes the full brain.

    Parameters
    ----------
    params : np.ndarray
        Array containing the parameters of the fit for the whole brain.

    Returns
    -------
    microFA : np.ndarray
        MicroFA values for the whole brain.
    MK_I : np.ndarray
        Isotropic mean kurtosis values for the whole brain.
    MK_A : np.ndarray
        Anisotropic mean kurtosis values for the whole brain.
    MK_T : np.ndarray
        Total mean kurtosis values for the whole brain.
    """
    # S0 = params[..., 0]
    MD = params[..., 1]
    V_I = params[..., 2]
    V_A = params[..., 3]
    V_T = V_I + V_A
    V_L = 5 / 2. * V_A

    MK_I = 3 * V_I / (MD ** 2)
    MK_A = 3 * V_A / (MD ** 2)
    MK_T = 3 * V_T / (MD ** 2)
    microFA2 = (3/2.) * (V_L / (V_I + V_L + (MD ** 2)))
    microFA = np.real(np.sqrt(microFA2))
    microFA[np.isnan(microFA)] = 0

    return microFA, MK_I, MK_A, MK_T


def _fit_gamma_parallel(args):
    data = args[0]
    gtab_infos = args[1]
    fit_iters = args[2]
    random_iters = args[3]
    do_weight_bvals = args[4]
    do_weight_pa = args[5]
    do_multiple_s0 = args[6]
    chunk_id = args[7]

    sub_fit_array = np.zeros((data.shape[0], 4))
    for i in range(data.shape[0]):
        if data[i].any():
            sub_fit_array[i] = _gamma_data2fit(data[i], gtab_infos, fit_iters,
                                               random_iters, do_weight_bvals,
                                               do_weight_pa, do_multiple_s0)

    return chunk_id, sub_fit_array


def fit_gamma(data, gtab_infos, mask=None, fit_iters=1, random_iters=50,
              do_weight_bvals=False, do_weight_pa=False, do_multiple_s0=False,
              nbr_processes=None):
    """Fit the gamma model to data

    Parameters
    ----------
    data : np.ndarray (4d)
        Diffusion data, powder averaged. Obtained as output of the function
        `reconst.b_tensor_utils.generate_powder_averaged_data`.
    gtab_infos : np.ndarray
        Contains information about the gtab, such as the unique bvals, the
        encoding types, the number of directions and the acquisition index.
        Obtained as output of the function
        `reconst.b_tensor_utils.generate_powder_averaged_data`.
    mask : np.ndarray, optional
        If `mask` is provided, only the data inside the mask will be
        used for computations.
    fit_iters : int, optional
        Number of iterations in the gamma fit. Defaults to 1.
    random_iters : int, optional
        Number of random sets of parameters tested to find the initial
        parameters. Defaults to 50.
    do_weight_bvals : bool , optional
        If set, does a weighting on the bvalues in the gamma fit.
    do_weight_pa : bool, optional
        If set, does a powder averaging weighting in the gamma fit.
    do_multiple_s0 : bool, optional
        If set, takes into account multiple baseline signals.
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

    nbr_processes = multiprocessing.cpu_count() if nbr_processes is None \
        or nbr_processes <= 0 else nbr_processes

    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    data = data[mask].reshape((np.count_nonzero(mask), data_shape[3]))
    chunks = np.array_split(data, nbr_processes)

    chunk_len = np.cumsum([0] + [len(c) for c in chunks])
    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(_fit_gamma_parallel,
                       zip(chunks,
                           itertools.repeat(gtab_infos),
                           itertools.repeat(fit_iters),
                           itertools.repeat(random_iters),
                           itertools.repeat(do_weight_bvals),
                           itertools.repeat(do_weight_pa),
                           itertools.repeat(do_multiple_s0),
                           np.arange(len(chunks))))
    pool.close()
    pool.join()

    # Re-assemble the chunk together in the original shape.
    fit_array = np.zeros((data_shape[0:3])+(4,))
    tmp_fit_array = np.zeros((np.count_nonzero(mask), 4))
    for i, fit in results:
        tmp_fit_array[chunk_len[i]:chunk_len[i+1]] = fit

    fit_array[mask] = tmp_fit_array

    return fit_array
