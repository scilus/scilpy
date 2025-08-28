""" Compute the Image Intra-class Correlation Coefficient (I2C2)^[1] for a series of subjects, each with multiple images (e.g., test-retest scans). The I2C2 is a measure of reliability that quantifies the proportion of variance in the images.

Each subject should have, for each timepoint, a metric image (e.g., FA, MD, etc.) and a ROI or label image. The I2C2 is computed within the ROI/labels. The output is a JSON file containing the I2C2 value and its confidence interval for each ROI/label.

Example usage:

    scil_volume_i2c2.py i2c2.json --subject metric1.nii.gz roi1.nii.gz metric2.nii.gz roi2.nii.gz \
            --subject metric1.nii.gz roi1.nii.gz metric2.nii.gz roi2.nii.gz

References:
    [1]: Shou, H., et al. (2013). Quantifying the reliability of image replication studies: the image intraclass correlation coefficient (I2C2). Cognitive, Affective, & Behavioral Neuroscience, 13(4), 714-724. https://doi.org/10.3758/s13415-013-0196-0
    """  # noqa: E501


import argparse
import json
import logging

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_json_args, add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_headers_compatible)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('out',
                   help='Path to the output JSON file to store I2C2 results.')
    p.add_argument('--subject', nargs='+', action='append', required=True,
                   help='Subject ID followed by pairs of metric and ROI files')
    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def load_data(subjects, label):
    """ Load and mask data from metric and ROI files for all subjects and visits.

    Need to load every ROI first to compute the union of all ROIs.

    Parameters
    ----------
    subjects : list of list of tuples
        Each element corresponds to a subject and contains a list of tuples.
        Each tuple contains (metric_file, roi_file) for a visit.
    label : int
        The label value in the ROI image to use for masking.
    Returns
    -------
    y : ndarray of shape (n_samples, n_features)
        Data matrix where each row corresponds to one observation
        (subject × visit) and each column is a voxel/feature within the ROI.
    id : ndarray of shape (n_samples,)
        Subject IDs corresponding to each observation.
    visit : ndarray of shape (n_samples,)
        Visit labels corresponding to each observation.
    """

    all_data = []
    all_ids = []
    all_visits = []
    all_rois = []

    # First pass: load all ROIs to compute union
    for subject_idx, metrics_rois in enumerate(subjects):
        for visit_idx, (metric_file, roi_file) in enumerate(metrics_rois):
            roi_img = nib.load(roi_file)
            roi_data = roi_img.get_fdata()
            all_rois.append(roi_data == label)
    union_roi = (np.sum(all_rois, axis=0) > 0).astype(np.uint8)

    # Second pass: load metric data and apply union ROI mask
    for subject_idx, metrics_rois in enumerate(subjects):
        subject_id = subject_idx + 1  # 1-based indexing
        for visit_idx, (metric_file, roi_file) in enumerate(metrics_rois):
            visit_label = visit_idx + 1  # 1-based indexing

            metric_img = nib.load(metric_file)
            metric_data = metric_img.get_fdata()

            if metric_data.shape != union_roi.shape:
                raise ValueError(f"Metric image {metric_file} and ROI image {roi_file} have different shapes.")

            masked_data = metric_data[union_roi > 0]  # Flatten and keep as row vector
            all_data.append(masked_data)
            all_ids.append(subject_id)
            all_visits.append(visit_label)

    y = np.asarray(all_data)
    id = np.array(all_ids)
    visit = np.array(all_visits)
    return y, id, visit

def I2C2_original(y,
                  id=None,
                  visit=None,
                  symmetric=False,
                  trun=False,
                  twoway=True,
                  demean=True):
    """
    Compute the Image Intraclass Correlation Coefficient (I2C2).

    This is a NumPy-only implementation of I2C2 (Shou et al., 2013).
    It quantifies the reliability of repeated imaging measurements.

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

    trun : bool, default=False
        If True, truncate negative I2C2 values to 0.

    twoway : bool, default=True
        If True, subtract both:
            - global mean (across all subjects/visits)
            - visit-specific deviations (removes scanner/batch effects)

    demean : bool, default=True
        If True, return demeaned data matrix in output.

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
    Shou, H.; Eloyan, A.; Lee, S.; Zipunnikov, V.; Crainiceanu, A.N.;
    Nebel, N.B.; Caffo, B.; Lindquist, M.A.; Crainiceanu, C.M. (2013).
    "Quantifying the reliability of image replication studies:
    the image intraclass correlation coefficient (I2C2)."
    Cogn Affect Behav Neurosci, 13(4):714–724.
    """

    # Ensure y is a numeric numpy array
    y = np.asarray(y, dtype=float)
    n, p = y.shape

    if id is None or visit is None:
        raise ValueError("You must provide both `id` and `visit` vectors.")

    # Reset IDs to consecutive integers (1..I)
    _, id = np.unique(id, return_inverse=True)
    id = id + 1  # convert from 0-based to 1-based indexing

    # -----------------------------------------------------------
    # Step 1: Demeaning (remove mean effects if requested)
    # -----------------------------------------------------------
    if demean:
        # Global mean across all observations
        mu = np.mean(y, axis=0)
        resd = y - mu

        if twoway:
            visit = np.asarray(visit)
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
    I = len(unique_ids)
    n_I0 = counts             # number of visits per subject
    k2 = np.sum(n_I0 ** 2)    # used in symmetric formula
    Wdd = np.mean(W, axis=0)  # global mean of W

    # Subject-specific sums
    Si = np.zeros((I, p))
    for i, uid in enumerate(unique_ids):
        mask = (id == uid)
        Si[i, :] = np.sum(W[mask, :], axis=0)

    # -----------------------------------------------------------
    # Step 3: Variance decomposition
    # -----------------------------------------------------------
    if not symmetric:
        # Method-of-moments estimator
        Wi = Si / n_I0[:, None]       # subject means
        Wi_expanded = Wi[id - 1, :]   # match each row to subject mean
        trKu = np.sum((W - Wi_expanded) ** 2) / (n - I)   # within-subject variance
        trKw = np.sum((W - Wdd) ** 2) / (n - 1)           # total variance
        trKx = trKw - trKu                               # between-subject variance
    else:
        # Pairwise symmetric sum estimator
        trKu = (np.sum(W ** 2 * n_I0[id - 1, None]) - np.sum(Si ** 2)) / (k2 - n)
        trKw = (np.sum(W ** 2) * n - np.sum((n * Wdd) ** 2) - trKu * (k2 - n)) / (n ** 2 - k2)
        trKx = trKw - trKu

    # -----------------------------------------------------------
    # Step 4: Compute I2C2
    # -----------------------------------------------------------
    lam = trKx / (trKx + trKu)
    if trun:
        lam = max(lam, 0)

    # -----------------------------------------------------------
    # Step 5: Return results
    # -----------------------------------------------------------
    result = {"lambda": lam, "Kx": trKx, "Ku": trKu}
    if demean:
        result["demean_y"] = W

    return result


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # For every subject, verify that we have pairs of metric and ROI files.
    # Verify that all files exist and that headers are compatible.
    subjects = []
    for subject in args.subject:
        metric_files = subject[0::2]
        roi_files = subject[1::2]
        assert_inputs_exist(parser, metric_files + roi_files)

        metrics_rois = list(zip(metric_files, roi_files))

        subjects.append(metrics_rois)

        for m, r in metrics_rois:
            assert_headers_compatible(parser, [m, r])

    # Compute I2C2
    all_results = {}
    for i in range(1, 10):
        data, ids, visits = load_data(subjects, label=i)
        i2c2_results = I2C2_original(data, ids, visits)['lambda']
        all_results[f'{i}'] = i2c2_results
    print(all_results)


if __name__ == "__main__":
    main()
