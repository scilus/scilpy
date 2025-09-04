import argparse
import json
import logging

import nibabel as nib
import numpy as np

from scilpy.image.volume_metrics import ICC_CV
from scilpy.io.utils import (add_json_args, add_overwrite_arg,
                             add_processes_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist,
                             assert_headers_compatible)
from scilpy.version import version_string


def _compute_icc_cv_labels(subjects):
    """ Load and mask data from metric and ROI files for all subjects and
    visits. Need to load every ROI first to compute the union of all ROIs
    to ensure consistent voxels across subjects/visits.

    Parameters
    ----------
    subjects : list of list of tuples
        Each element corresponds to a subject and contains a list of tuples.
        Each tuple contains (metric_file, roi_file) for a visit.
    mask_type : {'union', 'majority'}, optional
        Type of mask to use when combining ROIs across subjects and visits.
        'union' creates a mask that includes all voxels present in any ROI.
        'majority' creates a mask that includes voxels present in more than
        half of the ROIs. Default is 'union'.

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
    all_masks = []
    labels = set()

    # First pass: load all ROIs to get all labels
    for subject_idx, metrics_rois in enumerate(subjects):
        for visit_idx, (_, roi_file) in enumerate(metrics_rois):
            roi_img = nib.load(roi_file)
            roi_data = roi_img.get_fdata()
            all_masks.append(roi_data)
            labels.update(np.unique(roi_data).astype(int)[1:])

    labels = sorted(labels)

    # For each label, compute union ROI and load metric data
    for label in labels:

        icc_cv = ICC_CV(subjects, label)

        yield icc_cv, label


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('out',
                   help='Path to the output JSON file to store I2C2 results.')
    p.add_argument('--subject', nargs='+', action='append', required=True,
                   help='Subject ID followed by pairs of metric and ROI files')

    add_json_args(p)
    add_overwrite_arg(p)
    add_processes_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_outputs_exist(parser, args, args.out)

    # For every subject, verify that we have pairs of metric and ROI files.
    # Verify that all files exist and that headers are compatible.
    subjects = []
    for subject in args.subject:
        metric_files = subject[0::2]
        roi_files = subject[1::2]

        # Assert that each subject has atleast two timepoints
        if len(metric_files) < 2 or len(roi_files) < 2:
            parser.error('Each subject must have at least two timepoints.')

        assert_inputs_exist(parser, metric_files + roi_files)

        metrics_rois = list(zip(metric_files, roi_files))
        subjects.append(metrics_rois)

        for m, r in metrics_rois:
            assert_headers_compatible(parser, [m, r])

    # Compute I2C2
    all_results = {}
    for icc_cv, label in _compute_icc_cv_labels(subjects):

        all_results[f'{label}'] = icc_cv

    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
