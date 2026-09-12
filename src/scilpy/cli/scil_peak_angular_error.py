#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare two peaks images. The script reports the angular error between
estimated and reference peaks. The angular error is computed as the maximum angular
error between each reference peak and the closest estimated peak. A peak found in
the reference but not in the estimated peaks will contribute to the angular error,
but a peak found in the estimated peaks but not in the reference will not contribute
to the angular error.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_headers_compatible, assert_inputs_exist,
                             assert_outputs_exist, add_json_args)
from scilpy.version import version_string
import json


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_peaks',
                   help='Path of the input peaks image.')
    p.add_argument('in_peaks_ref',
                   help='Path to the reference peaks image against which to compare.')
    p.add_argument('out_angular_error',
                   help='Output filename for the angular error map between estimated and reference peaks.')
    p.add_argument('out_json',
                   help='Output JSON file to save the computed metrics.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_json_args(p)
    return p


def _compute_angular_error(est_peaks, ref_peaks):
    """
    Angular error between estimated and reference peaks. The angular error
    is computed as the maximum angular error between each reference peak and
    the closest estimated peak. A peak found in the reference but not in the
    estimated peaks will contribute to the angular error, but a peak found in
    the estimated peaks but not in the reference will not contribute to the
    angular error.
    """
    angular_error = np.zeros(est_peaks.shape[:3])

    # iterate over reference peaks
    for peak_i in range(ref_peaks.shape[-2]):
        ref_peaks_i = ref_peaks[..., peak_i, :]
        valid_i = np.linalg.norm(ref_peaks_i, axis=-1) > 0
        if not np.any(valid_i):
            continue  # no valid peaks in whole image, skip to next peak

        # compare the current reference peak to all estimated peaks
        dot_prod = np.abs(
            np.sum(est_peaks * ref_peaks_i[..., None, :], axis=-1))
        dot_prod = np.clip(dot_prod, -1, 1)  # clip for numerical stability

        # then the angular error for this peak is the smallest angle
        # between the reference peak and all estimated peaks
        _angular_error = np.rad2deg(np.arccos(dot_prod))
        # take the minimum error across all estimated peaks
        _angular_error = np.min(_angular_error, axis=-1)
        # ignore voxels without valid peaks in the reference
        _angular_error[~valid_i] = 0
        # take the maximum error across all peaks
        angular_error = np.maximum(angular_error, _angular_error)

    return angular_error


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_peaks, args.in_peaks_ref])
    assert_outputs_exist(parser, args, [args.out_json, args.out_angular_error])

    assert_headers_compatible(parser, [args.in_peaks, args.in_peaks_ref])

    # Load reference peaks and normalize
    ref_peaks_im = nib.load(args.in_peaks_ref)
    ref_peaks = ref_peaks_im.get_fdata(dtype=np.float32)
    ref_peaks = ref_peaks.reshape(ref_peaks.shape[:-1] + (-1, 3))
    norm = np.linalg.norm(ref_peaks, axis=-1)
    ref_peaks = np.divide(ref_peaks, norm[..., None],
                          where=norm[..., None] > 0)

    # create a nufo map for the reference peaks
    valid = norm > 0  # valid is 4D
    # 3D mask of voxels with at least one valid peak
    mask = np.sum(valid, axis=-1) > 0
    nufo = np.count_nonzero(valid, axis=-1)
    nufo[~mask] = 0

    # Load estimated peaks
    est_peaks_im = nib.load(args.in_peaks)
    est_peaks = est_peaks_im.get_fdata(dtype=np.float32)
    est_peaks = est_peaks.reshape(est_peaks.shape[:3] + (-1, 3))
    # normalize estimated peaks to unit length
    norm = np.linalg.norm(est_peaks, axis=-1, keepdims=True)
    est_peaks = np.divide(est_peaks, norm, where=norm > 0)

    # compute the maximum angular error between estimated and reference peaks
    max_angular_error = _compute_angular_error(est_peaks, ref_peaks)

    # metrics for whole image
    metrics_dict = {}
    metrics_dict['mean_max_angular_error'] = np.mean(max_angular_error[mask])
    for i in range(1, nufo.max() + 1):
        count = np.count_nonzero(nufo == i)
        if count > 0:
            metrics_dict[f'mean_max_angular_error_nufo_{i}'] =\
                np.mean(max_angular_error[nufo == i])

    # Save outputs
    nib.save(nib.Nifti1Image(max_angular_error.astype(np.float32), ref_peaks_im.affine),
             args.out_angular_error)

    # JSON metrics
    with open(args.out_json, 'w') as outfile:
        json.dump(metrics_dict, outfile,
                  indent=args.indent,
                  sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
