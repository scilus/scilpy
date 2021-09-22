import logging

from dipy.core.gradients import (gradient_table,
                                 unique_bvals_tolerance, get_bval_indices)
from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.utils.bvec_bval_tools import (normalize_bvecs, is_normalized_bvecs,
                                          extract_dwi_shell)


bshapes = {0: "STE", 1: "LTE", -0.5: "PTE", 0.5: "CTE"}


def generate_btensor_input(input_files, bvals_files, bvecs_files,
                           b_deltas_list, force_b0_threshold,
                           do_pa_signals=False, tol=20):
    data_full = np.empty(0)
    bvals_full = np.empty(0)
    bvecs_full = np.empty(0)
    b_deltas = np.empty(0)
    b_shapes = np.empty(0)
    ubvals_full = np.empty(0)
    ub_deltas_full = np.empty(0)
    nb_bvecs_full = np.empty(0)
    acq_index_full = np.empty(0)
    ubvals_divide = np.empty(0)

    acq_index = 0
    for inputf, bvalsf, bvecsf, b_delta in zip(input_files, bvals_files,
                                               bvecs_files, b_deltas_list):
        if inputf:
            vol = nib.load(inputf)
            # data = vol.get_fdata(dtype=np.float32)
            bvals, bvecs = read_bvals_bvecs(bvalsf, bvecsf)
            if np.sum([bvals > tol]) != 0:
                bvals = np.round(bvals)
            if not is_normalized_bvecs(bvecs):
                logging.warning('Your b-vectors do not seem normalized...')
                bvecs = normalize_bvecs(bvecs)
            # check_b0_threshold(force_b0_threshold, bvals.min())

            ubvals = unique_bvals_tolerance(bvals, tol=tol)
            for ubval in ubvals:  # Loop over all unique bvals
                # Extracting the data for the shell ubval
                indices, shell_data, _, output_bvecs = \
                    extract_dwi_shell(vol, bvals, bvecs, [ubval], tol=tol)
                nb_bvecs = len(indices)
                acq_index_full = np.concatenate([acq_index_full,
                                                 [acq_index]]) \
                    if acq_index_full.size else np.array([acq_index])
                ubvals_divide = np.concatenate([ubvals_divide, [ubval]]) \
                    if ubvals_divide.size else np.array([ubval])
                same_bvals = np.argwhere(ubvals_full == ubval)
                # Dealing with ubvals, ub_deltas and nb_bvecs for b0 or not
                if (same_bvals.size  # Differenciate bvals over bdeltas
                   and b_delta != ub_deltas_full[same_bvals]):
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
                b_deltas = np.concatenate([b_deltas, np.repeat([b_delta],
                                                               nb_bvecs)]) \
                    if b_deltas.size else np.repeat([b_delta], nb_bvecs)
                b_shapes = np.concatenate([b_shapes,
                                           np.repeat([bshapes[b_delta]],
                                                     nb_bvecs)]) \
                    if b_shapes.size else np.repeat([bshapes[b_delta]],
                                                    nb_bvecs)
            acq_index += 1

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

    duplicate_b0_ind = np.union1d(np.argwhere(ubvals_full == 0),
                                  np.argwhere(ubvals_full > tol))
    ubvals_full = ubvals_full[duplicate_b0_ind]
    ub_deltas_full = ub_deltas_full[duplicate_b0_ind]

    # Sorting the data by bvals
    sorted_indices = np.argsort(bvals_full, axis=0)
    bvals_full = np.take_along_axis(bvals_full, sorted_indices, axis=0)
    bvals_full[bvals_full < tol] = 0
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
