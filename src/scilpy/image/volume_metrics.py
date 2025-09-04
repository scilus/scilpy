# -*- coding: utf-8 -*-
import logging

import numpy as np
import nibabel as nib
import pandas as pd

from multiprocessing import Pool

from dipy.denoise.noise_estimate import piesno
from statsmodels.regression.mixed_linear_model import MixedLM


def estimate_piesno_sigma(data, number_coils=0):
    """
    Here are Dipy's note on this method:
    > It is expected that
    >   1. The data has a noisy, non-masked background and
    >   2. The data is a repetition of the same measurements along the last
    >      axis, i.e. dMRI or fMRI data, not structural data like T1/T2.
    > This function processes the data slice by slice, as originally designed
    > inthe paper. Use it to get a slice by slice estimation of the noise, as
    > in spinal cord imaging for example.

    Parameters
    ----------
    data: np.ndarray
        The 4D volume.
    number_coils: int
        The number of coils in the scanner.

    Returns
    -------
    sigma: np.ndarray
        The piesno sigma estimation, one per slice.
    mask_noise: np.ndarray
        The mask of pure noise voxels inferred from the data.
    """
    assert len(data.shape) == 4

    sigma, mask_noise = piesno(data, N=number_coils, return_mask=True)

    # If the noise mask has few voxels, the detected noise standard
    # deviation can be very low and maybe something went wrong. We
    # check here that at least 1% of noisy voxels were found and warn
    # the user otherwise.
    frac_noisy_voxels = np.sum(mask_noise) / np.size(mask_noise) * 100

    if frac_noisy_voxels < 1.:
        logging.warning(
            'PIESNO was used with N={}, but it found only {:.3f}% of voxels '
            'as pure noise with a mean standard deviation of {:.5f}. This is '
            'suspicious, so please check the resulting sigma volume if '
            'something went wrong. In cases where PIESNO is not working, '
            'you might want to try basic sigma estimation.'
            .format(number_coils, frac_noisy_voxels, np.mean(sigma)))
    else:
        logging.info('The noise standard deviation from piesno is %s',
                     np.array_str(sigma[0, 0, :]))

    return sigma, mask_noise


def I2C2(
    y,
    id=None,
    visit=None,
    symmetric=False,
    truncate=False,
    twoway=True,
    demean=True
):
    """
    Compute the Image Intraclass Correlation Coefficient (I2C2).

    This is a numpy re-implementation of I2C2[1] (Shou et al., 2013)[2].
    It quantifies the reliability of repeated imaging measurements.

    Directly adapted from:
        https://github.com/muschellij2/I2C2/blob/master/R/I2C2_orig.R

    translated by ChatGPT, then further adapted.

    Parameters
    ----------
    y : ndarray of shape (n, p)
        Data matrix where each row corresponds to one observation
        (e.g., subject × visit) and each column is a voxel/feature.
        - n = total number of observations (subjects × visits)
        - p = number of voxels

    id : ndarray of shape (n,)
        Subject IDs. Example: [1,1,2,2,3,3] for 3 subjects with 2 visits each.

    visit : ndarray of shape (n,)
        Visit labels. Example: [1,2,1,2,1,2].

    symmetric : bool, default=False
        If False, use method-of-moments estimator.
        If True, use pairwise symmetric sum estimator.

    truncate : bool, default=False
        If True, truncate negative I2C2 values to 0.

    twoway : bool, default=True
        If True, subtract both:
            - global mean (across all subjects/visits)
            - visit-specific deviations (removes scanner/batch effects)

    Returns
    -------
    result : dict
        Dictionary with the following keys:
        - 'lambda' : float
            Estimated I2C2 value (ratio of between-subject to total variance).
        - 'Kx' : float
            Trace of between-subject variance component.
        - 'Ku' : float
            Trace of within-subject variance component.
        - 'demean_y' : ndarray of shape (n, p), optional
            Demeaned data matrix (returned only if demean=True).

    References
    ----------
    [1]: https://github.com/muschellij2/I2C2
    [2]: Shou, H.; Eloyan, A.; Lee, S.; Zipunnikov, V.; Crainiceanu, A.N.;
    Nebel, N.B.; Caffo, B.; Lindquist, M.A.; Crainiceanu, C.M. (2013).
    "Quantifying the reliability of image replication studies:
    the image intraclass correlation coefficient (I2C2)."
    Cogn Affect Behav Neurosci, 13(4):714–724.
    """

    # Ensure y is a numeric numpy array
    n, p = y.shape

    # -----------------------------------------------------------
    # Step 1: Demeaning (remove mean effects if requested)
    # -----------------------------------------------------------
    if demean:
        # Global mean across all observations
        mu = np.mean(y, axis=0)
        resd = y - mu

        if twoway:
            unique_visits = np.unique(visit)
            eta = np.zeros((len(unique_visits), p))

            # Compute visit-specific deviations from global mean
            for idx, v in enumerate(unique_visits):
                mask = (visit == v)
                if np.sum(mask) == 1:
                    eta[idx, :] = y[mask, :].reshape(-1) - mu
                else:
                    eta[idx, :] = np.mean(y[mask, :], axis=0) - mu

            # Subtract visit-specific means
            for idx, v in enumerate(unique_visits):
                mask = (visit == v)
                resd[mask, :] = y[mask, :] - (mu + eta[idx, :])

        W = resd  # demeaned dataset
    else:
        W = y

    # -----------------------------------------------------------
    # Step 2: Compute subject-specific statistics
    # -----------------------------------------------------------
    unique_ids, counts = np.unique(id, return_counts=True)
    I = len(unique_ids)  # noqa E741
    J = counts             # number of visits per subject
    k2 = np.sum(J ** 2)    # used in symmetric formula
    Wdd = np.mean(W, axis=0)  # global mean of W

    # Subject-specific sums
    Si = np.zeros((I, p))
    for i, uid in enumerate(unique_ids):
        sub_mask = (id == uid)
        Si[i, :] = np.sum(W[sub_mask, :], axis=0)

    # -----------------------------------------------------------
    # Step 3: Variance decomposition
    # -----------------------------------------------------------
    if not symmetric:
        # Method-of-moments estimator
        Wi = Si / J[:, None]       # subject means
        Wi_expanded = Wi[id, :]   # match each row to subject mean
        trKu = np.sum((W - Wi_expanded) ** 2) / (n - I)  # within-subject var
        trKw = np.sum((W - Wdd) ** 2) / (n - 1)          # total variance
        trKx = trKw - trKu                               # between-subject var
    else:
        # Pairwise symmetric sum estimator
        trKu = (
            np.sum(W ** 2 * J[id, None]) - np.sum(Si ** 2)) / (k2 - n)
        trKw = (
            np.sum(W ** 2) * n - np.sum((n * Wdd) ** 2) - trKu * (k2 - n)
        ) / (n ** 2 - k2)

    trKx = trKw - trKu  # between-subject var

    # -----------------------------------------------------------
    # Step 4: Compute I2C2
    # -----------------------------------------------------------

    # Check if trKx + trKu is zero to avoid division by zero
    if trKx + trKu == 0:
        lam = 1.0
    else:
        lam = trKx / (trKx + trKu)
        if truncate:
            lam = max(lam, 0)

    # -----------------------------------------------------------
    # Step 5: Return results
    # -----------------------------------------------------------
    result = {"lambda": lam}

    return result


def _single_resample(args):
    y, id, visit, symmetric, truncate, twoway, demean = args

    # Resample subjects with replacement
    unique_subj = np.unique(id)
    n_subj = unique_subj.shape[0]

    masks = []
    for _ in range(n_subj):
        subj = np.random.choice(unique_subj)
        mask = (id == subj)
        masks.extend(mask)

    # Perform logical OR to combine masks
    masks = np.array(masks).reshape(n_subj, -1)
    resampled_indices = np.sum(masks, axis=0).astype(bool)

    # Create resampled dataset
    y_res = y[resampled_indices, :]
    id_res = id[resampled_indices]
    visit_res = visit[resampled_indices]

    # Change subject IDs to be consecutive integers
    unique_resubj = np.unique(id_res)
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_resubj)}
    id_res = np.array([id_map[old_id] for old_id in id_res])

    # Compute I2C2
    return I2C2(y_res, id=id_res, visit=visit_res,
                symmetric=symmetric, truncate=truncate,
                twoway=twoway, demean=demean)['lambda']


def I2C2_mcCI(y, id, visit, R=100, seed=None, ci=0.95, processes=1,
              symmetric=False, truncate=False, twoway=True, demean=True):
    """
    Compute bootstrap confidence interval for I2C2 using Monte Carlo resampling
    Also returns the original I2C2 estimate.

    Parameters
    ----------
    y : ndarray (n, p)
        Input data matrix (observations × features).
    id : ndarray (n,)
        Subject identifiers for each observation.
    visit : ndarray (n,)
        Visit labels for each observation.
    R : int, default=100
        Number of bootstrap replications.
    seed : int or None
        Random seed for reproducibility.
    ci : float, default=0.95
        Confidence level (e.g., 0.95 for a 95% CI).
    mc_cores : int, default=1
        Number of parallel processes to use.
    symmetric: bool, default=False
        If False, use method-of-moments estimator.
        If True, use pairwise symmetric sum estimator.
    truncate : bool, default=False
        If True, truncate negative I2C2 values to 0.
    twoway : bool, default=True
        If True, subtract both global and visit-specific means.
    demean : bool, default=True
        If True, demean data before I2C2 computation.

    Returns
    -------
    dict with keys:
        'lambdas' : ndarray of shape (R,)
            Bootstrap I2C2 estimates.
        'ci_lower' : float
            Lower bound of the confidence interval.
        'ci_upper' : float
            Upper bound of the confidence interval.
    """

    lamb = I2C2(y, id=id, visit=visit,
                symmetric=symmetric, truncate=truncate,
                twoway=twoway, demean=demean)['lambda']

    if seed is not None:
        np.random.seed(seed)
    args = (y, id, visit, symmetric, truncate, twoway, demean)
    # Parallel or serial execution
    if processes > 1:
        with Pool(processes) as pool:
            lambdas = np.array(pool.map(_single_resample,
                                        [args for r in range(R)]))
    else:
        lambdas = np.array([_single_resample(args) for r in range(R)])

    # Compute CI bounds
    alpha = (1 - ci) / 2
    ci_lower = np.quantile(lambdas, alpha)
    ci_upper = np.quantile(lambdas, 1 - alpha)

    return {'lambda': lamb, 'ci_lower': ci_lower, 'ci_upper': ci_upper}


def _compute_icc_lme(data: pd.DataFrame, roi: str):
    """
    Compute ICC using a linear mixed-effects model
    Model: value ~ 1 + (1 | subject)
    """

    # Mixed model with random intercept for subject
    # Raises a lot of warnings, can be ignored
    model = MixedLM(data["data"], np.ones(len(data)), groups=data["id"])
    result = model.fit(maxiter=500, reml=True, full_output=True)

    logging.debug(result.summary())

    # Variance components
    var_subject = result.cov_re.iloc[0, 0]  # between-subject variance
    var_resid = result.scale  # residual (within-subject) variance

    icc = var_subject / (var_subject + var_resid)
    return icc


def _compute_cv(data: pd.DataFrame):
    """
    Compute mean within-subject coefficient of variation for a given ROI.
    """
    cvs = []
    for subj, group in data.groupby("id"):
        vals = group["data"].values
        if vals.mean() != 0 and len(vals) > 1:
            cv = vals.std(ddof=1) / vals.mean()
            cvs.append(cv)
    return np.mean(cvs) if cvs else np.nan


def ICC_CV(subjects, label):
    """
    Compute the Coefficient of Variation (CV) and Intraclass Correlation
    Coefficient (ICC) for a set of subjects with repeated measurements.

    The ICC and CV are computed for a specific ROI label across all subjects
    for the mean metric values within that ROI.

    Parameters
    ----------
    subjects : list of list of 2-tuples
        Each subject is represented as a list of (metric_file, roi_file) tuples
        Example: [[(metric1_s1, roi1_s1), (metric2_s1, roi2_s1)],  # Subject 1
                  [(metric1_s2, roi1_s2), (metric2_s2, roi2_s2)],  # Subject 2
                  ...]
        Each subject must have at least two timepoints.
    label : int
        The ROI label to compute metrics for.

    Returns
    -------
    results : dict
        Dictionary with keys as ROI labels and values as another dict with:
        - 'CV': Coefficient of Variation
        - 'ICC': Intraclass Correlation Coefficient
    """

    all_data = []
    all_ids = []

    # Load metric data and apply union ROI mask
    for sub_id, metrics_rois in enumerate(subjects):
        for metric_file, roi_file in metrics_rois:

            metric_img = nib.load(metric_file)
            metric_data = metric_img.get_fdata()

            roi_img = nib.load(roi_file)
            roi_data = roi_img.get_fdata()

            # Flatten and mask data
            data = metric_data[roi_data == label]
            if data.size > 0:
                mean = data.mean() * 100  # Scale to avoid numerical issues
            else:
                mean = 0

            all_data.append(mean)
            all_ids.append(sub_id)

    # Convert to a pandas DataFrame for easier manipulation
    df = pd.DataFrame({'id': all_ids, 'data': all_data})

    # Compute CV
    cv = _compute_cv(df)
    # Compute ICC
    icc = _compute_icc_lme(df, label)

    return {'CV': cv, 'ICC': icc}
