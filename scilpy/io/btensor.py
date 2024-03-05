import logging

from dipy.core.gradients import (gradient_table,
                                 unique_bvals_tolerance, get_bval_indices)
from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.dwi.utils import extract_dwi_shell
from scilpy.gradients.bvec_bval_tools import (normalize_bvecs,
                                              is_normalized_bvecs,
                                              check_b0_threshold)


bshapes = {0: "STE", 1: "LTE", -0.5: "PTE", 0.5: "CTE"}
bdeltas = {"STE": 0, "LTE": 1, "PTE": -0.5, "CTE": 0.5}


def convert_bshape_to_bdelta(b_shapes):
    """Convert an array of b_shapes to an array of b_deltas.

    Parameters
    ----------
    b_shapes: array of strings
        b_shapes to convert. Strings can only be LTE, PTE, STE or CTE.

    Returns
    -------
    b_deltas: array of floats
        Converted b_deltas, such that LTE = 1, STE = 0, PTE = -0.5, CTE = 0.5.
    """
    b_deltas = np.vectorize(bdeltas.get)(b_shapes)
    return b_deltas


def convert_bdelta_to_bshape(b_deltas):
    """Convert an array of b_deltas to an array of b_shapes.

    Parameters
    ----------
    b_deltas: array of floats
        b_deltas to convert. Floats can only be 1, 0, -0.5 or 0.5.

    Returns
    -------
    b_shapes: array of strings
        Converted b_shapes, such that LTE = 1, STE = 0, PTE = -0.5, CTE = 0.5.
    """
    b_shapes = np.vectorize(bshapes.get)(b_deltas)
    return b_shapes


def generate_btensor_input(in_dwis, in_bvals, in_bvecs, in_bdeltas,
                           do_pa_signals=False, tol=20, skip_b0_check=False):
    """Generate b-tensor input from an ensemble of data, bvals and bvecs files.
    This generated input is mandatory for all scripts using b-tensor encoding
    data. Also generate the powder-averaged (PA) data if set.

    For the moment, this does not enable the use of a b0_threshold different
    than the tolerance.

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
    do_pa_signals : bool, optional
        If set, will compute the powder_averaged input instead of the regular
        one. This means that the signal is averaged over all directions for
        each bvals.
    tol : int
        tolerance gap for b-values clustering. Defaults to 20
    skip_b0_check: bool
        (See full explanation in io.utils.add_skip_b0_check_arg.) If true,
        script will continue even if no b-values are found below the tolerance
        (no b0s found).

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
            _ = check_b0_threshold(bvals.min(), b0_thr=tol,
                                   skip_b0_check=skip_b0_check)
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
        return pa_signals, gtab_infos
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

    return gtab_full, data_full, ubvals_full, ub_deltas_full
