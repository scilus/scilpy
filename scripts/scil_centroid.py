#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
import nibabel as nib
from nibabel.streamlines.tractogram import Tractogram
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute bundle centroid',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('bundle',
                   help='Fiber bundle file.')
    p.add_argument('centroid_streamline',
                   help='Output centroid streamline file.')

    p.add_argument('--distance_thres',
                   type=float, default=200,
                   help='The maximum distance from a bundle for a streamline '
                        'to be still considered as part of it')
    p.add_argument('--nb_points',
                   type=int, default=20,
                   help='Number of points defining the centroid streamline')

    add_overwrite_arg(p)
    return p


def get_centroid_streamline(tractogram, nb_points, distance_threshold):
    streamlines = tractogram.streamlines
    resample_feature = ResampleFeature(nb_points=nb_points)
    quick_bundle = QuickBundles(
        threshold=distance_threshold,
        metric=AveragePointwiseEuclideanMetric(resample_feature))
    clusters = quick_bundle.cluster(streamlines)
    centroid_streamlines = clusters.centroids

    if len(centroid_streamlines) > 1:
        raise Exception('Multiple centroids found')

    return Tractogram(centroid_streamlines, affine_to_rasmm=np.eye(4))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.bundle)
    assert_outputs_exists(parser, args, args.centroid_streamline)

    if args.distance_thres < 0.0:
        parser.error('--distance_thres {} should be '
                     'positive'.format(args.distance_thres))
    if args.nb_points < 2:
        parser.error('--nb_points {} should be >= 2'
                     .format(args.nb_points))

    tractogram = nib.streamlines.load(args.bundle)
    centroid_streamline = get_centroid_streamline(tractogram,
                                                  args.nb_points,
                                                  args.distance_thres)
    nib.streamlines.save(centroid_streamline,
                         args.centroid_streamline, header=tractogram.header)


if __name__ == '__main__':
    main()
