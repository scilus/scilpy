#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remove outliers from image using the RANSAC algorithm.
The RANSAC algorithm parameters are sensitive to the input data.

NOTE: Current default parameters are tuned for ad/md/rd images only.

Formerly: scil_remove_outliers_ransac.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np
from sklearn import linear_model

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_image',
                   help='Nifti image.')
    p.add_argument('out_image',
                   help='Corrected Nifti image.')

    p.add_argument('--min_fit', type=int, default=50,
                   help='The minimum number of data values required ' +
                        'to fit the model. [%(default)s]')
    p.add_argument('--max_iter', type=int, default=1000,
                   help='The maximum number of iterations allowed ' +
                        'in the algorithm. [%(default)s]')
    p.add_argument('--fit_thr', type=float, default=1e-2,
                   help='Threshold value for determining when a data ' +
                        'point fits a model. [%(default)s]')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image)
    assert_outputs_exist(parser, args, args.out_image)

    if args.min_fit < 2:
        parser.error('--min_fit should be at least 2. Current value: {}'
                     .format(args.min_fit))
    if args.max_iter < 1:
        parser.error('--max_iter should be at least 1. Current value: {}'
                     .format(args.max_iter))
    if args.fit_thr <= 0:
        parser.error('--fit_thr should be greater than 0. Current value: {}'
                     .format(args.fit_thr))

    in_img = nib.load(args.in_image)
    in_data = in_img.get_fdata(dtype=np.float32)

    if np.average(in_data[in_data > 0]) > 0.1:
        logging.warning('Be careful, your image doesn\'t seem to be an ad, '
                        'md or rd.')

    in_data_flat = in_data.flatten()
    in_nzr_ind = np.nonzero(in_data_flat)
    in_nzr_val = np.array(in_data_flat[in_nzr_ind])

    X = in_nzr_ind[0][:, np.newaxis]
    model_ransac = linear_model.RANSACRegressor(
        base_estimator=linear_model.LinearRegression(),
        min_samples=args.min_fit,
        residual_threshold=args.fit_thr,
        max_trials=args.max_iter)
    model_ransac.fit(X, in_nzr_val)

    outlier_mask = np.logical_not(model_ransac.inlier_mask_)
    outliers = X[outlier_mask]

    logging.info('# outliers: {}'.format(len(outliers)))

    in_data_flat[outliers] = 0

    out_data = np.reshape(in_data_flat, in_img.shape)
    nib.save(nib.Nifti1Image(out_data, in_img.affine, in_img.header),
             args.out_image)


if __name__ == '__main__':
    main()
