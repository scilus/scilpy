import logging

from dipy.core.gradients import (gradient_table,
                                 unique_bvals_tolerance, get_bval_indices)
from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.utils.bvec_bval_tools import (check_b0_threshold, normalize_bvecs,
                                          is_normalized_bvecs)


bshapes = {0: "STE", 1: "LTE", -0.5: "PTE", 0.5: "CTE"}


def generate_btensor_input(input_files, bvals_files, bvecs_files,
                           b_deltas_list, force_b0_threshold, tol=20):
    data_conc = []
    data_b0 = []
    bvals_conc = []
    bvals_b0 = []
    bvecs_conc = []
    bvecs_b0 = []
    b_deltas = []
    b_shapes = []
    ubvals_conc = []
    ub_deltas_conc = []
    for inputf, bvalsf, bvecsf, b_delta in zip(input_files, bvals_files,
                                               bvecs_files, b_deltas_list):
        if inputf is not None:
            vol = nib.load(inputf)
            data = vol.get_fdata(dtype=np.float32)
            bvals, bvecs = read_bvals_bvecs(bvalsf, bvecsf)
            if np.sum([bvals > tol]) != 0:
                bvals = np.round(bvals)

            if not is_normalized_bvecs(bvecs):
                logging.warning('Your b-vectors do not seem normalized...')
                bvecs = normalize_bvecs(bvecs)

            # check_b0_threshold(force_b0_threshold, bvals.min())

            ubvals = unique_bvals_tolerance(bvals, tol=tol)
            for ubval in ubvals:
                indices = get_bval_indices(bvals, ubval, tol=tol)
                if ubval < tol:
                    if data_b0 == []:
                        data_b0 = data[..., indices]
                        bvals_b0 = np.zeros(len(indices))
                        bvecs_b0 = bvecs[indices]
                        ubvals_conc = [0]
                        ub_deltas_conc = [1]
                    else:
                        data_b0 = np.concatenate((data_b0, data[..., indices]),
                                                 axis=-1)
                        bvals_b0 = np.concatenate((bvals_b0,
                                                   np.zeros(len(indices))))
                        bvecs_b0 = np.concatenate((bvecs_b0, bvecs[indices]))
                        if np.sum([ubvals_conc < tol]) < 1:
                            ubvals_conc = np.concatenate((ubvals_conc, [0]))
                            ub_deltas_conc = np.concatenate((ub_deltas_conc,
                                                             [1]))
                else:
                    if data_conc == []:
                        data_conc = data[..., indices]
                        bvals_conc = np.ones(len(indices)) * ubval
                        bvecs_conc = bvecs[indices]
                        b_deltas = np.ones(len(indices)) * b_delta
                        b_shapes = np.repeat(np.array([bshapes[b_delta]]),
                                             len(indices))
                    else:
                        data_conc = np.concatenate((data_conc,
                                                    data[..., indices]),
                                                    axis=-1)
                        bvals_conc = np.concatenate((bvals_conc,
                                                     np.ones(len(indices)) * ubval))
                        bvecs_conc = np.concatenate((bvecs_conc,
                                                     bvecs[indices]))
                        b_deltas = np.concatenate((b_deltas,
                                                   np.ones(len(indices)) * b_delta))
                        b_shapes = np.concatenate((b_shapes,
                                                   np.repeat(np.array([bshapes[b_delta]]),
                                                                      len(indices))))
                    ubvals_conc = np.concatenate((ubvals_conc, [ubval]))
                    ub_deltas_conc = np.concatenate((ub_deltas_conc, [b_delta]))

    data_conc = np.concatenate((data_b0, data_conc), axis=-1)
    bvals_conc = np.concatenate((bvals_b0, bvals_conc))
    bvecs_conc = np.concatenate((bvecs_b0, bvecs_conc))
    b_deltas = np.concatenate((np.ones(len(bvals_b0)), b_deltas))
    b_shapes = np.concatenate((np.repeat(np.array(["LTE"]), len(bvals_b0)),
                               b_shapes))

    ubvals = []
    ub_deltas = []
    while len(ubvals_conc):
        bval = ubvals_conc[0]
        bdelta = ub_deltas_conc[0]
        indices = get_bval_indices(ubvals_conc, bval, tol=tol)
        if len(indices) > 1:
            ubvals.append(bval)
            ub_deltas.append(bdelta)
            for i in indices[1:]:
                if ubvals_conc[i] == bval:
                    ubvals_conc[i] += 1
                    bvals_conc = np.where(np.logical_and(bvals_conc == bval,
                                          b_deltas != ub_deltas_conc[indices[0]]),
                                          bval + 1, bvals_conc)
            ubvals_conc = np.delete(ubvals_conc, indices[0])
            ub_deltas_conc = np.delete(ub_deltas_conc, indices[0])
        else:
            ubvals.append(bval)
            ub_deltas.append(bdelta)
            ubvals_conc = np.delete(ubvals_conc, indices)
            ub_deltas_conc = np.delete(ub_deltas_conc, indices)

    sorted_indices = np.argsort(bvals_conc, axis=0)
    bvals_conc = np.take_along_axis(bvals_conc, sorted_indices, axis=0)
    bvecs_conc = np.take_along_axis(bvecs_conc,
                                    sorted_indices.reshape(len(bvals_conc), 1),
                                    axis=0)
    b_shapes = np.take_along_axis(b_shapes, sorted_indices, axis=0)
    data_conc = np.take_along_axis(data_conc,
                                   sorted_indices.reshape(1, 1, 1, len(bvals_conc)),
                                   axis=-1)

    sorted_indices = np.argsort(np.asarray(ubvals), axis=0)
    ubvals = np.take_along_axis(np.asarray(ubvals), sorted_indices, axis=0)
    ub_deltas = np.take_along_axis(np.asarray(ub_deltas), sorted_indices, axis=0)

    gtab_conc = gradient_table(bvals_conc, bvecs_conc,
                               b0_threshold=bvals_conc.min(),
                               btens=b_shapes)

    return(gtab_conc, data_conc, ubvals, ub_deltas)


def generate_powder_averaged_data(input_files, bvals_files, bvecs_files,
                                  b_deltas_list, force_b0_threshold, tol=20):
    pa_signals_conc = []
    ub_deltas_conc = []
    ubvals_conc = []
    nb_ubvecs_conc = []
    acq_index_conc = []

    b0_conc = []

    acq_index_current = 0
    for inputf, bvalsf, bvecsf, b_delta in zip(input_files, bvals_files,
                                               bvecs_files, b_deltas_list):
        if inputf is not None:
            vol = nib.load(inputf)
            data = vol.get_fdata(dtype=np.float32)
            bvals, bvecs = read_bvals_bvecs(bvalsf, bvecsf)

            if np.sum([bvals > tol]) != 0:
                bvals = np.round(bvals)

            # check_b0_threshold(force_b0_threshold, bvals.min())

            ubvals = unique_bvals_tolerance(bvals, tol=tol)
            # ubvals = ubvals[ubvals > tol]
            pa_signals = np.zeros(((data.shape[:-1])+(len(ubvals),)))
            ub_deltas = np.ones(len(ubvals)) * b_delta
            nb_ubvecs = np.zeros(len(ubvals))
            acq_index = np.ones(len(ubvals)) * acq_index_current
            for i, ubval in enumerate(ubvals):
                indices = get_bval_indices(bvals, ubval, tol=tol)
                pa_signals[..., i] = np.nanmean(data[..., indices], axis=-1)
                nb_ubvecs[i] = len(indices)

            if pa_signals_conc == []:
                pa_signals_conc = pa_signals
                ub_deltas_conc = ub_deltas
                ubvals_conc = ubvals
                nb_ubvecs_conc = nb_ubvecs
                acq_index_conc = acq_index
            else:
                pa_signals_conc = np.concatenate((pa_signals_conc, pa_signals),
                                                 axis=-1)
                ub_deltas_conc = np.concatenate((ub_deltas_conc, ub_deltas))
                ubvals_conc = np.concatenate((ubvals_conc, ubvals))
                nb_ubvecs_conc = np.concatenate((nb_ubvecs_conc, nb_ubvecs))
                acq_index_conc = np.concatenate((acq_index_conc, acq_index))

            acq_index_current += 1

    pa_signals = np.asarray(pa_signals_conc)

    gtab_infos = np.ndarray((4, len(ubvals_conc)))
    gtab_infos[0] = ubvals_conc
    gtab_infos[1] = ub_deltas_conc
    gtab_infos[2] = nb_ubvecs_conc
    gtab_infos[3] = acq_index_conc

    if np.sum([ubvals_conc < tol]) < acq_index_current - 1:
        gtab_infos[3] *= 0

    return(pa_signals, gtab_infos)


def extract_affine(input_files):
    for input_file in input_files:
        if input_file is not None:
            vol = nib.load(input_file)
            return vol.get_affine()
