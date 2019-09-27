#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_reference)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute bundle centroid',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file.')

    add_reference(p)

    p.add_argument('out_centroid',
                   help='Output centroid streamline filename.')
    p.add_argument('--distance_thres',
                   type=float, default=200,
                   help='The maximum distance from a bundle for a streamline '
                        'to be still considered as part of it.')
    p.add_argument('--nb_points',
                   type=int, default=20,
                   help='Number of points defining the centroid streamline.')

    add_overwrite_arg(p)
    return p


def get_centroid_streamline(streamlines, nb_points, distance_threshold):
    resample_feature = ResampleFeature(nb_points=nb_points)
    quick_bundle = QuickBundles(
        threshold=distance_threshold,
        metric=AveragePointwiseEuclideanMetric(resample_feature))
    clusters = quick_bundle.cluster(streamlines)
    centroid_streamlines = clusters.centroids

    if len(centroid_streamlines) > 1:
        raise Exception('Multiple centroids found')

    return centroid_streamlines


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundle)
    assert_outputs_exist(parser, args, args.out_centroid)

    if args.distance_thres < 0.0:
        parser.error('--distance_thres {} should be '
                     'positive'.format(args.distance_thres))
    if args.nb_points < 2:
        parser.error('--nb_points {} should be >= 2'
                     .format(args.nb_points))

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)

    centroid_streamlines = get_centroid_streamline(sft.streamlines,
                                                   args.nb_points,
                                                   args.distance_thres)

    sft = StatefulTractogram(centroid_streamlines, sft, Space.RASMM)

    save_tractogram(sft, args.out_centroid)


if __name__ == '__main__':
    main()
