#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Segment a single bundle by computing a simple Recobundles (single-atlas &
single-parameters).

For multiple bundles segmentation (using RecobundlesX / BundleSeg), see instead
>>> scil_tractogram_segment_with_bundleseg.py

Hints:
- The model needs to be cleaned and lightweight.
- The transform should come from ANTs: (using the --inverse flag)
  >>> AntsRegistrationSyNQuick.sh -d 3 -m MODEL_REF -f SUBJ_REF
  If you are unsure about the transformation 'direction', try with and without
  the --inverse flag. If you are not using the right transformation 'direction'
  a warning will pop up. If there is no warning in both cases, it means the
  transformation is very close to identity and both 'directions' will work.

Formerly: scil_recognize_single_bundles.py
-------------------------------------------------------------------------------
Reference:
[1] Garyfallidis, E., Cote, M. A., Rheault, F., ... & Descoteaux, M. (2018).
    Recognition of white matter bundles using local and global streamline-based
    registration and clustering. NeuroImage, 170, 283-295.
-------------------------------------------------------------------------------
"""

import argparse
import logging
import pickle

from dipy.segment.bundles import RecoBundles
from dipy.tracking.streamline import transform_streamlines
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np

from scilpy.io.streamlines import (load_tractogram_with_reference,
                                   save_tractogram)
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, load_matrix_in_any_format,
                             ranged_type)
from scilpy.utils.spatial import compute_distance_barycenters
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractogram',
                   help='Input tractogram filename.')
    p.add_argument('in_model',
                   help='Model bundle to use for recognition. (Ex, a .trk '
                        'file.')
    p.add_argument('in_transfo',
                   help='Path for the transformation to model space '
                        '(.txt, .npy or .mat).')
    p.add_argument('out_tractogram',
                   help='Output tractogram filename.')

    g = p.add_argument_group("Recobundles options")
    g.add_argument('--tractogram_clustering_thr',
                   type=ranged_type(float, 0, None),
                   help='Clustering threshold used for the whole brain. '
                        'Default: 8.')
    g.add_argument('--model_clustering_thr',
                   type=ranged_type(float, 0, None), default=4,
                   help='Clustering threshold used for the model '
                        '[%(default)smm].')
    g.add_argument('--pruning_thr',
                   type=ranged_type(float, 0, None), default=6,
                   help='MDF threshold used for final streamlines selection '
                        '[%(default)smm].')
    g.add_argument('--slr_threads', type=int, default=1,
                   help='Number of threads for SLR [%(default)s].')

    p.add_argument('--seed', type=int, default=None,
                   help='Random number generator seed [%(default)s].')
    p.add_argument('--inverse', action='store_true',
                   help='Use the inverse transformation.')
    p.add_argument('--no_empty', action='store_true',
                   help='Do not write file if there is no streamline.')

    group = p.add_mutually_exclusive_group()
    group.add_argument('--in_pickle',
                       help='Input pickle clusters map file.\nWill override '
                            'the tractogram_clustering_thr parameter.')
    group.add_argument('--out_pickle',
                       help='Output pickle clusters map file.')

    add_reference_arg(p, 'in_tractogram')
    add_reference_arg(p, 'in_model')
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, [args.in_tractogram, args.in_transfo],
                        [args.in_tractogram_ref, args.in_model_ref,
                         args.in_pickle])
    assert_outputs_exist(parser, args, args.out_tractogram,
                         args.out_pickle)

    if args.tractogram_clustering_thr and args.in_pickle:
        parser.error("Option --tractogram_clustering_thr should not be "
                     "used with --in_pickle.")
    else:
        # Setting default value. (Will be ignored in args.in_pickle)
        args.tractogram_clustering_thr = 8.0

    # Loading
    wb_sft = load_tractogram_with_reference(parser, args, args.in_tractogram,
                                            arg_name='in_tractogram')
    model_sft = load_tractogram_with_reference(parser, args, args.in_model,
                                               arg_name='in_model')
    transfo = load_matrix_in_any_format(args.in_transfo)

    # Processing
    if args.inverse:
        transfo = np.linalg.inv(transfo)

    # Dipy's method below use RASMM space, CENTER origin. This is the default.
    # But, just to be sure:
    wb_sft.to_rasmm()
    wb_sft.to_center()
    model_sft.to_rasmm()
    model_sft.to_center()

    # 1) Register the streamlines to the model space (apply transform)
    before, after = compute_distance_barycenters(wb_sft, model_sft, transfo)
    if after > before:
        logging.warning('The distance between volumes barycenter should be '
                        'lower after registration. Maybe try using/removing '
                        '--inverse.')
        logging.info('Distance before: {}, Distance after: {}'.format(
            np.round(before, 3), np.round(after, 3)))
    model_streamlines = transform_streamlines(model_sft.streamlines, transfo)

    # 2) Prepare Recobundles object
    rng = np.random.RandomState(args.seed)
    verbose = args.verbose in ['DEBUG', 'INFO']
    if args.in_pickle:
        with open(args.in_pickle, 'rb') as infile:
            cluster_map = pickle.load(infile)
        reco_obj = RecoBundles(wb_sft.streamlines,
                               cluster_map=cluster_map,
                               rng=rng, less_than=1,
                               verbose=verbose)
    else:
        reco_obj = RecoBundles(wb_sft.streamlines,
                               clust_thr=args.tractogram_clustering_thr,
                               rng=rng, greater_than=1,
                               verbose=verbose)

    if args.out_pickle:
        logging.info("Saving out_pickle")
        with open(args.out_pickle, 'wb') as outfile:
            pickle.dump(reco_obj.cluster_map, outfile)

    # 3) Run recobundle.
    _, indices = reco_obj.recognize(ArraySequence(model_streamlines),
                                    args.model_clustering_thr,
                                    pruning_thr=args.pruning_thr,
                                    num_threads=args.slr_threads)

    # Save results
    bundle_sft = wb_sft[indices]
    save_tractogram(bundle_sft, args.out_tractogram, args.no_empty)


if __name__ == '__main__':
    main()
