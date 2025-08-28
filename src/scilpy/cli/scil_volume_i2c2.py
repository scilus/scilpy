""" Compute the Image Intra-class Correlation Coefficient (I2C2)^[1] for a series of subjects, each with multiple images (e.g., test-retest scans). The I2C2 is a measure of reliability that quantifies the proportion of variance in the images.

Each subject should have, for each timepoint, a metric image (e.g., FA, MD, etc.) and a ROI or label image. The I2C2 is computed within the ROI/labels. The output is a JSON file containing the I2C2 value and its confidence interval for each ROI/label.

Example usage:

    scil_volume_i2c2.py i2c2.json --subject metric1.nii.gz roi1.nii.gz metric2.nii.gz roi2.nii.gz \
            --subject metric1.nii.gz roi1.nii.gz metric2.nii.gz roi2.nii.gz \
            --subject ...

References:
    [1]: Shou, H., et al. (2013). Quantifying the reliability of image replication studies: the image intraclass correlation coefficient (I2C2). Cognitive, Affective, & Behavioral Neuroscience, 13(4), 714-724. https://doi.org/10.3758/s13415-013-0196-0
    """  # noqa: E501


import argparse
import json
import logging

import nibabel as nib
import numpy as np

from scilpy.image.volume_metrics import I2C2
from scilpy.io.utils import (add_json_args, add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_headers_compatible)
from scilpy.version import version_string


def load_data(subjects, label):
    """ Load and mask data from metric and ROI files for all subjects and
    visits. Need to load every ROI first to compute the union of all ROIs
    to ensure consistent voxels across subjects/visits.

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
        subject_id = subject_idx
        for visit_idx, (metric_file, roi_file) in enumerate(metrics_rois):
            visit_label = visit_idx

            metric_img = nib.load(metric_file)
            metric_data = metric_img.get_fdata()

            if metric_data.shape != union_roi.shape:
                raise ValueError(f"Metric image {metric_file} and ROI image "
                                 "{roi_file} have different shapes.")

            masked_data = metric_data[union_roi > 0].flatten()
            all_data.append(masked_data)
            all_ids.append(subject_id)
            all_visits.append(visit_label)

    y = np.asarray(all_data)
    id = np.array(all_ids)
    visit = np.array(all_visits)
    return y, id, visit


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('out',
                   help='Path to the output JSON file to store I2C2 results.')
    p.add_argument('--subject', nargs='+', action='append', required=True,
                   help='Subject ID followed by pairs of metric and ROI files')
    p.add_argument('--symmetric', action='store_true',
                   help='Use pairwise symmetric sum estimator '
                        '(default: method-of-moments).')
    p.add_argument('--truncate', action='store_true',
                   help='Truncate negative I2C2 values to 0 (default: allow '
                        'negative values).')
    p.add_argument('--no-twoway', action='store_false', dest='twoway',
                   help='Do not remove visit-specific means (default: remove '
                        'both global and visit-specific means).')
    p.add_argument('--no-demean', action='store_false', dest='demean',
                   help='Do not demean data (default: demean data).')
    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


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
        i2c2_results = I2C2(data, ids, visits,
                            symmetric=args.symmetric,
                            trun=args.truncate,
                            twoway=args.twoway,
                            demean=args.demean)
        all_results[f'{i}'] = i2c2_results
    print(all_results)

    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
