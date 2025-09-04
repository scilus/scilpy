""" Compute ICC and CV for each label in the provided ROIs across multiple subjects and visits. ICC and CV are computed on the mean value within each ROI.

This script can compare multiple subjects each with a different number of timepoints. Each subject should have, for each timepoint, a metric image (e.g., FA, MD, etc.) and a ROI or label image. ICC and CV are computed within the ROI/labels. The output is a JSON file containing the ICC and CV for each ROI/label.

Example usage (for 2 subjects, one with 2 visits, the other with 3 visits, using the CC as ROI and FA as metric):

    scil_volume_icc_cv icc_cv.json --subject sub-1_ses-1__fa.nii.gz sub-1_ses-1__CC.nii.gz sub-1_ses-2__fa.nii.gz sub-1_ses-2__CC.nii.gz \
            --subject  sub-2_ses-1__fa.nii.gz sub-2_ses-1__CC.nii.gz sub-2_ses-2__fa.nii.gz sub-2_ses-2__CC.nii.gz sub-2_ses-3__fa.nii.gz sub-2_ses-3__CC.nii.gz

    As shown above, the --subject argument should be followed by pairs of metric and ROI files for each visit. Multiple --subject arguments can be provided for different subjects.
    """  # noqa: E501

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


def _compute_icc_cv_labels(subjects, twoway):
    """ Load and mask data from metric and ROI files for all subjects and
    visits. 

    Parameters
    ----------
    subjects : list of list of tuples
        Each element corresponds to a subject and contains a list of tuples.
        Each tuple contains (metric_file, roi_file) for a visit.
    twoway : bool
        If True, use two-way mixed effects model for ICC calculation.
        Default is one-way random effects model.

    Returns
    -------
    generator of (icc_cv, label) tuples
        icc_cv is a dict with keys 'ICC' and 'CV' containing the computed
        values for the given label.
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

        icc_cv = ICC_CV(subjects, label, twoway=twoway)

        yield icc_cv, label


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('out',
                   help='Path to the output JSON file to store I2C2 results.')
    p.add_argument('--subject', nargs='+', action='append', required=True,
                   help='Subject ID followed by pairs of metric and ROI files')
    p.add_argument('--twoway', action='store_true',
                   help='Use two-way mixed effects model for ICC calculation. '
                   'Default is one-way random effects model.')

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

        if len(metric_files) != len(roi_files):
            parser.error('Each subject must have the same number of metric '
                         'and ROI/labels files.')

        # Assert that each subject has atleast two timepoints
        if len(metric_files) < 2 or len(roi_files) < 2:
            parser.error('Each subject must have at least two timepoints.')

        assert_inputs_exist(parser, metric_files + roi_files)

        metrics_rois = list(zip(metric_files, roi_files))
        subjects.append(metrics_rois)

        for m, r in metrics_rois:
            assert_headers_compatible(parser, [m, r])

    # If twoway is specified, ensure that each subject has the same number
    # of visits
    if args.twoway:
        n_visits = len(subjects[0])
        for metrics_rois in subjects:
            if len(metrics_rois) != n_visits:
                parser.error('When using --twoway, all subjects must have '
                             'the same number of visits.')

    # Compute ICC and CV for each label
    all_results = {}
    for icc_cv, label in _compute_icc_cv_labels(subjects, args.twoway):

        all_results[f'{label}'] = icc_cv

    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
