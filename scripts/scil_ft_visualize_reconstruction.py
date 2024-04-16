#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize reconstruction of fibertubes resulting from fibertube tracking.
(See scil_ft_tracking.py)
"""
import argparse
import logging
import numpy as np
import nibabel as nib

from fury import window, actor
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             add_reference_arg,
                             load_dictionary,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_centroids',
                   help='Path to the tractogram file containing the \n'
                   'fibertubes\' centroids (must be .trk or .tck).')

    p.add_argument('in_diameters',
                   help='Path to a text file containing a list of the \n'
                   'diameters of each fibertube in mm (.txt). Each line \n'
                   'corresponds to the identically numbered centroid.')

    p.add_argument('in_tractogram',
                   help='Tractogram file containing the ground truth fiber \n'
                   'reconstruction (must be .trk or .tck).')

    p.add_argument('in_config',
                   help='Path to a text file containing the fibertube \n'
                   'parameters used for the tracking process.')

    p.add_argument('--single_diameter', action='store_true',
                   help='If set, the first diameter found in \n'
                   '[in_diameters] will be repeated for each fiber.')

    p.add_argument('--save',
                   help='If set, save a screenshot of the result in the \n'
                   'specified filename')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_reference_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, [args.in_centroids, args.in_diameters,
                                 args.in_tractogram])
    assert_outputs_exist(parser, args, [], [args.save])

    if not nib.streamlines.is_supported(args.in_centroids):
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk or tck): {0}".format(
                             args.in_centroids))

    if not nib.streamlines.is_supported(args.in_tractogram):
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk or tck): {0}".format(
                             args.in_centroids))

    logging.debug('Loading centroid tractogram & diameters')
    truth_sft = load_tractogram_with_reference(parser, args, args.in_centroids)
    truth_sft.to_voxmm()
    truth_sft.to_center()
    # Casting ArraySequence as a list to improve speed
    fibers = list(truth_sft.get_streamlines_copy())
    diameters = np.loadtxt(args.in_diameters, dtype=np.float64)
    if args.single_diameter:
        diameters = [diameters[0]]*len(fibers)

    logging.debug('Loading reconstruction tractogram')
    in_sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    in_sft.to_voxmm()
    in_sft.to_center()

    logging.debug("Loading config")
    config = load_dictionary(args.in_config)
    nb_fibers = int(config['nb_fibers'])
    nb_seeds_per_fiber = int(config['nb_seeds_per_fiber'])

    # Make display objects and add them to canvas
    s = window.Scene()
    for i, line in enumerate(truth_sft.streamlines[:nb_fibers]):
        truth_actor = actor.streamtube([line], colors=[1., 0., 0.],
                                       opacity=0.8, linewidth=diameters[i])
        s.add(truth_actor)

    nb_streamlines = nb_fibers*nb_seeds_per_fiber
    in_actor = actor.line(in_sft.streamlines[:nb_streamlines],
                          colors=[0., 1., 0.])
    s.add(in_actor)

    # Allow to zoom enough to at least see something
    s.SetClippingRangeExpansion(0)

    # Show and record if needed
    if args.save is not None:
        window.record(s, out_path=args.save, size=(1000, 1000))
    window.show(s)


if __name__ == '__main__':
    main()
