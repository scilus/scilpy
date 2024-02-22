# -*- coding: utf-8 -*-

import logging

import numpy as np


def create_acqparams(readout, encoding_direction, synb0=False,
                     nb_b0s=1, nb_rev_b0s=1):
    """
    Create acqparams for Topup and Eddy

    Parameters
    ----------
    readout: float
        Readout time
    encoding_direction: string
        Encoding direction (x, y or z)
    nb_b0s: int
        Number of B=0 images
    nb_rev_b0s: int
        number of reverse b=0 images

    Returns
    -------
    acqparams: np.array
        acqparams
    """
    if synb0:
        logging.warning('Using SyNb0, untested feature. Be careful.')

    acqparams = np.zeros((nb_b0s + nb_rev_b0s, 4))
    acqparams[:, 3] = readout

    enum_direction = {'x': 0, 'y': 1, 'z': 2}
    acqparams[0:nb_b0s, enum_direction[encoding_direction]] = 1
    if nb_rev_b0s > 0:
        val = -1 if not synb0 else 1
        acqparams[nb_b0s:, enum_direction[encoding_direction]] = val
        acqparams[nb_b0s:, 3] = readout if not synb0 else 0

    return acqparams


def create_index(bvals, n_rev=0):
    """
    Create index of bvals for Eddy

    Parameters
    ----------
    bvals: np.array
        b-values
    n_rev: int, optional
        Number of reverse phase images to take into account

    Returns
    -------
    index: np.array
    """
    index = np.ones(len(bvals), dtype=int)
    index[len(index)-n_rev:] += 1

    return index.tolist()


def create_multi_topup_index(bvals, mean, n_rev, b0_thr=0):
    """
    Create index of bvals for Eddy in cases where Topup ran on more
    than one b0 volume in both phase directions. The volumes must be
    ordered such as all forward phase acquisition are followed by all
    reverse phase ones (In the case of AP-PA, PA_1, PA_2, ..., PA_N,
    AP_1, AP_2, ..., AP_N).

    Parameters
    ----------
    bvals: np.array
        b-values
    mean: string
        Mean strategy used to subset the b0 volumes
        passed to topup (cluster or none)
    n_rev: int, optional
        Number of reverse phase images to take into account
    b0_thr: int
        All bvals under or equal to this threshold are considered as b0

    Returns
    -------
    index: np.array
    """
    index = np.zeros_like(bvals, dtype=int)
    cnt = 1

    mask = np.ma.masked_array(bvals, bvals <= b0_thr)
    whole_b0_clumps = [list(np.ma.clump_masked(mask[:-n_rev]))]
    whole_dw_clumps = [list(np.ma.clump_unmasked(mask[:-n_rev]))]

    if n_rev > 0:
        n_for = len(bvals) - n_rev
        whole_b0_clumps += [[slice(s.start + n_for, s.stop + n_for)
                            for s in list(np.ma.clump_masked(mask[n_rev:]))]]
        whole_dw_clumps += [[slice(s.start + n_for, s.stop + n_for)
                            for s in list(np.ma.clump_unmasked(mask[n_rev:]))]]

    for b0_clumps, dw_clumps in zip(whole_b0_clumps, whole_dw_clumps):
        if b0_clumps[0].start > dw_clumps[0].start:
            index[dw_clumps[0]] = 1
            dw_clumps = dw_clumps[1:]

        for s1, s2 in zip(b0_clumps[:len(dw_clumps)], dw_clumps):
            if mean == "none":
                index[s1] = np.arange(cnt, cnt + s1.stop - s1.start)
                index[s2] = index[s1.stop - 1]
                cnt += s1.stop - s1.start
            elif mean == "cluster":
                index[s1] = index[s2] = cnt
                cnt += 1
            else:
                raise ValueError('Undefined mean category for '
                                 'index determination : {}'.format(mean))

        if len(b0_clumps) > len(dw_clumps):
            if mean == "none":
                index[b0_clumps[-1]] = np.arange(
                    cnt, cnt + b0_clumps[-1].stop - b0_clumps[-1].start)
                cnt += b0_clumps[-1].stop - b0_clumps[-1].start
            elif mean == "cluster":
                index[b0_clumps[-1]] = cnt
                cnt += 1
            else:
                raise ValueError('Undefined mean category for '
                                 'index determination : {}'.format(mean))

    return index


def create_non_zero_norm_bvecs(bvecs):
    """
    Add an epsilon to bvecs with a non zero norm.
    Mandatory for Topup and Eddy

    Parameters
    ----------
    bvecs: np.array
        b-vectors
    Returns
    -------
    bvecs: np.array
        b-vectors with an epsilon
    """
    # Set the bvecs to an epsilon if the norm is 0.
    # Mandatory to compute topup/eddy
    for i in range(len(bvecs)):
        if np.linalg.norm(bvecs[i, :]) < 0.00000001:
            bvecs[i, :] += 0.00000001

    return bvecs
