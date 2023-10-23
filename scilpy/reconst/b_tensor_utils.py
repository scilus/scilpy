import logging

from dipy.data import default_sphere
from dipy.core.gradients import (gradient_table, GradientTable,
                                 unique_bvals_tolerance, get_bval_indices)
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst import shm
from dipy.reconst.mcsd import MultiShellResponse
import nibabel as nib
import numpy as np

from scilpy.utils.bvec_bval_tools import (normalize_bvecs, is_normalized_bvecs,
                                          extract_dwi_shell)


bshapes = {0: "STE", 1: "LTE", -0.5: "PTE", 0.5: "CTE"}


def generate_btensor_input(in_dwis, in_bvals, in_bvecs,
                           in_bdeltas, force_b0_threshold,
                           do_pa_signals=False, tol=20):
    """Generate b-tensor input from an ensemble of data, bvals and bvecs files.
    This generated input is mandatory for all scripts using b-tensor encoding
    data. Also generate the powder-averaged (PA) data if set.

    Parameters
    ----------
    in_dwis : list of strings (file paths)
        Diffusion data files for each b-tensor encodings.
    in_bvals : list of strings (file paths)
        All of the bvals files associated.
    in_bvecs : list of strings (file paths)
        All of the bvecs files associated.
    in_bdeltas : list of floats
        All of the b_deltas (describing the type of encoding) files associated.
    force_b0_threshold : bool, optional
        If set, will continue even if the minimum bvalue is suspiciously high.
    do_pa_signals : bool, optional
        If set, will compute the powder_averaged input instead of the regular
        one. This means that the signal is averaged over all directions for
        each bvals.
    tol : int
        tolerance gap for b-values clustering. Defaults to 20

    Returns
    -------
    gtab_full : GradientTable
        A single gradient table containing the information of all encodings.
    data_full : np.ndarray (4d)
        All concatenated diffusion data from the different encodings.
    ubvals_full : array
        All the unique bvals from the different encodings, but with a single
        b0. If two or more encodings have the same bvalue, then they are
        differentiate by +1.
    ub_deltas_full : array
        All the b_delta values associated with `ubvals_full`.
    pa_signals : np.ndarray (4d) (if `do_pa_signals`)
        Powder-averaged diffusion data.
    gtab_infos : np.ndarray (if `do_pa_signals`)
        Contains information about the gtab, such as the unique bvals, the
        encoding types, the number of directions and the acquisition index.
    """
    data_full = np.empty(0)
    bvals_full = np.empty(0)
    bvecs_full = np.empty(0)
    b_shapes = np.empty(0)
    ubvals_full = np.empty(0)
    ub_deltas_full = np.empty(0)
    nb_bvecs_full = np.empty(0)
    acq_index_full = np.empty(0)
    ubvals_divide = np.empty(0)
    acq_index = 0
    for inputf, bvalsf, bvecsf, b_delta in zip(in_dwis, in_bvals,
                                               in_bvecs, in_bdeltas):
        if inputf:  # verifies if the input file exists
            vol = nib.load(inputf)
            bvals, bvecs = read_bvals_bvecs(bvalsf, bvecsf)
            if np.sum([bvals > tol]) != 0:
                bvals = np.round(bvals)
            if not is_normalized_bvecs(bvecs):
                logging.warning('Your b-vectors do not seem normalized...')
                bvecs = normalize_bvecs(bvecs)
            ubvals = unique_bvals_tolerance(bvals, tol=tol)
            for ubval in ubvals:  # Loop over all unique bvals
                # Extracting the data for the ubval shell
                indices, shell_data, _, output_bvecs = \
                    extract_dwi_shell(vol, bvals, bvecs, [ubval], tol=tol)
                nb_bvecs = len(indices)
                # Adding the current data to each arrays of interest
                acq_index_full = np.concatenate([acq_index_full,
                                                 [acq_index]]) \
                    if acq_index_full.size else np.array([acq_index])
                ubvals_divide = np.concatenate([ubvals_divide, [ubval]]) \
                    if ubvals_divide.size else np.array([ubval])
                while np.isin(ubval, ubvals_full):  # Differentiate the bvals
                    ubval += 1
                ubvals_full = np.concatenate([ubvals_full, [ubval]]) \
                    if ubvals_full.size else np.array([ubval])
                ub_deltas_full = np.concatenate([ub_deltas_full, [b_delta]]) \
                    if ub_deltas_full.size else np.array([b_delta])
                nb_bvecs_full = np.concatenate([nb_bvecs_full, [nb_bvecs]]) \
                    if nb_bvecs_full.size else np.array([nb_bvecs])
                data_full = np.concatenate([data_full, shell_data], axis=-1) \
                    if data_full.size else shell_data
                bvals_full = np.concatenate([bvals_full,
                                             np.repeat([ubval], nb_bvecs)]) \
                    if bvals_full.size else np.repeat([ubval], nb_bvecs)
                bvecs_full = np.concatenate([bvecs_full, output_bvecs]) \
                    if bvecs_full.size else output_bvecs
                b_shapes = np.concatenate([b_shapes,
                                           np.repeat([bshapes[b_delta]],
                                                     nb_bvecs)]) \
                    if b_shapes.size else np.repeat([bshapes[b_delta]],
                                                    nb_bvecs)
            acq_index += 1
    # In the case that the PA data is wanted, there is a different return
    if do_pa_signals:
        pa_signals = np.zeros(((data_full.shape[:-1])+(len(ubvals_full),)))
        for i, ubval in enumerate(ubvals_full):
            indices = get_bval_indices(bvals_full, ubval, tol=0)
            pa_signals[..., i] = np.nanmean(data_full[..., indices], axis=-1)
        gtab_infos = np.ndarray((4, len(ubvals_full)))
        gtab_infos[0] = ubvals_divide
        gtab_infos[1] = ub_deltas_full
        gtab_infos[2] = nb_bvecs_full
        gtab_infos[3] = acq_index_full
        if np.sum([ubvals_full < tol]) < acq_index - 1:
            gtab_infos[3] *= 0
        return(pa_signals, gtab_infos)
    # Removing the duplicate b0s from ubvals_full
    duplicate_b0_ind = np.union1d(np.argwhere(ubvals_full == min(ubvals_full)),
                                  np.argwhere(ubvals_full > tol))
    ubvals_full = ubvals_full[duplicate_b0_ind]
    ub_deltas_full = ub_deltas_full[duplicate_b0_ind]
    # Sorting the data by bvals
    sorted_indices = np.argsort(bvals_full, axis=0)
    bvals_full = np.take_along_axis(bvals_full, sorted_indices, axis=0)
    bvals_full[bvals_full < tol] = min(ubvals_full)
    bvecs_full = np.take_along_axis(bvecs_full,
                                    sorted_indices.reshape(len(bvals_full), 1),
                                    axis=0)
    b_shapes = np.take_along_axis(b_shapes, sorted_indices, axis=0)
    data_full = np.take_along_axis(data_full,
                                   sorted_indices.reshape(1, 1, 1,
                                                          len(bvals_full)),
                                   axis=-1)
    # Sorting the ubvals
    sorted_indices = np.argsort(np.asarray(ubvals_full), axis=0)
    ubvals_full = np.take_along_axis(np.asarray(ubvals_full), sorted_indices,
                                     axis=0)
    ub_deltas_full = np.take_along_axis(np.asarray(ub_deltas_full),
                                        sorted_indices, axis=0)
    # Creating the corresponding gtab
    gtab_full = gradient_table(bvals_full, bvecs_full,
                               b0_threshold=bvals_full.min(),
                               btens=b_shapes)

    return(gtab_full, data_full, ubvals_full, ub_deltas_full)


def single_tensor_btensor(gtab, evals, b_delta, S0=1):
    # This function should be moved to Dipy at some point

    if b_delta > 1 or b_delta < -0.5:
        msg = """The value of b_delta must be between -0.5 and 1."""
        raise ValueError(msg)

    out_shape = gtab.bvecs.shape[:gtab.bvecs.ndim - 1]
    gradients = gtab.bvecs.reshape(-1, 3)

    evals = np.asarray(evals)
    D_iso = np.sum(evals) / 3.
    D_para = evals[np.argmax(abs(evals - D_iso))]
    D_perp = evals[np.argmin(abs(evals - D_iso))]
    D_delta = (D_para - D_perp) / (3 * D_iso)

    S = np.zeros(len(gradients))
    for (i, g) in enumerate(gradients):
        theta = np.arctan2(np.sqrt(g[0] ** 2 + g[1] ** 2), g[2])
        P_2 = (3 * np.cos(theta) ** 2 - 1) / 2.
        b = gtab.bvals[i]
        S[i] = S0 * np.exp(-b * D_iso * (1 + 2 * b_delta * D_delta * P_2))

    return S.reshape(out_shape)


def multi_shell_fiber_response(sh_order, bvals, wm_rf, gm_rf, csf_rf,
                               b_deltas=None, sphere=None, tol=20):
    # This function should be moved to Dipy at some point

    bvals = np.array(bvals, copy=True)

    n = np.arange(0, sh_order + 1, 2)
    m = np.zeros_like(n)

    if sphere is None:
        sphere = default_sphere

    big_sphere = sphere.subdivide()
    theta, phi = big_sphere.theta, big_sphere.phi

    B = shm.real_sh_descoteaux_from_index(m, n, theta[:, None], phi[:, None])
    A = shm.real_sh_descoteaux_from_index(0, 0, 0, 0)

    if b_deltas is None:
        b_deltas = np.ones(len(bvals) - 1)

    response = np.empty([len(bvals), len(n) + 2])

    if bvals[0] < tol:
        gtab = GradientTable(big_sphere.vertices * 0)
        wm_response = single_tensor_btensor(gtab, wm_rf[0, :3], 1, wm_rf[0, 3])
        response[0, 2:] = np.linalg.lstsq(B, wm_response, rcond=None)[0]

        response[0, 1] = gm_rf[0, 3] / A
        response[0, 0] = csf_rf[0, 3] / A
        for i, bvalue in enumerate(bvals[1:]):
            gtab = GradientTable(big_sphere.vertices * bvalue)
            wm_response = single_tensor_btensor(gtab, wm_rf[i, :3],
                                                b_deltas[i],
                                                wm_rf[i, 3])
            response[i+1, 2:] = np.linalg.lstsq(B, wm_response, rcond=None)[0]

            response[i+1, 1] = gm_rf[i, 3] * np.exp(-bvalue * gm_rf[i, 0]) / A
            response[i+1, 0] = csf_rf[i, 3] * np.exp(-bvalue
                                                     * csf_rf[i, 0]) / A

        S0 = [csf_rf[0, 3], gm_rf[0, 3], wm_rf[0, 3]]

    else:
        logging.warning('No b0 was given. Proceeding either way.')
        for i, bvalue in enumerate(bvals):
            gtab = GradientTable(big_sphere.vertices * bvalue)
            wm_response = single_tensor_btensor(gtab, wm_rf[i, :3],
                                                b_deltas[i],
                                                wm_rf[i, 3])
            response[i, 2:] = np.linalg.lstsq(B, wm_response, rcond=None)[0]

            response[i, 1] = gm_rf[i, 3] * np.exp(-bvalue * gm_rf[i, 0]) / A
            response[i, 0] = csf_rf[i, 3] * np.exp(-bvalue * csf_rf[i, 0]) / A

        S0 = [csf_rf[0, 3], gm_rf[0, 3], wm_rf[0, 3]]

    return MultiShellResponse(response, sh_order, bvals, S0=S0)