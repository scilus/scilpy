#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a simple Recobundles (single-atlas & single-parameters).
The model need to be cleaned and lightweight.
Transform should come from ANTs: (using the --inverse flag)
AntsRegistration -m MODEL_REF -f SUBJ_REF
ConvertTransformFile 3 0GenericAffine.mat 0GenericAffine.npy --ras --hm
"""

import argparse
import pickle

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.segment.bundles import RecoBundles
from dipy.tracking.streamline import transform_streamlines
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)

EPILOG = """
Garyfallidis, E., Côté, M. A., Rheault, F., ... &
Descoteaux, M. (2018). Recognition of white matter
bundles using local and global streamline-based registration and
clustering. NeuroImage, 170, 283-295.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
        epilog=EPILOG)

    p.add_argument('in_tractogram',
                   help='Input tractogram filename.')
    p.add_argument('in_model',
                   help='Model to use for recognition.')
    p.add_argument('in_transfo',
                   help='Path for the transformation to model space '
                        '(.txt, .npy or .mat).')
    p.add_argument('out_tractogram',
                   help='Output tractogram filename.')

    p.add_argument('--tractogram_clustering_thr', type=float, default=8,
                   help='Clustering threshold used for the whole brain '
                        '[%(default)smm].')
    p.add_argument('--model_clustering_thr', type=float, default=4,
                   help='Clustering threshold used for the model '
                        '[%(default)smm].')
    p.add_argument('--pruning_thr', type=float, default=6,
                   help='MDF threshold used for final streamlines selection '
                        '[%(default)smm].')

    p.add_argument('--slr_threads', type=int, default=None,
                   help='Number of threads for SLR [all].')
    p.add_argument('--seed', type=int, default=None,
                   help='Random number generator seed [%(default)s].')
    p.add_argument('--inverse', action='store_true',
                   help='Use the inverse transformation.')
    p.add_argument('--no_empty', action='store_true',
                   help='Do not write file if there is no streamline.')

    group = p.add_mutually_exclusive_group()
    group.add_argument('--in_pickle',
                       help='Input pickle clusters map file.\n'
                            'Will override the tractogram_clustering_thr parameter.')
    group.add_argument('--out_pickle',
                       help='Output pickle clusters map file.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.in_transfo])
    assert_outputs_exist(parser, args, args.out_tractogram)

    wb_file = load_tractogram_with_reference(parser, args, args.in_tractogram)
    wb_streamlines = wb_file.streamlines
    model_file = load_tractogram_with_reference(parser, args, args.in_model)

    # Default transformation source is expected to be ANTs
    transfo = load_matrix_in_any_format(args.in_transfo)
    if args.inverse:
        transfo = np.linalg.inv(np.loadtxt(args.in_transfo))

    model_streamlines = transform_streamlines(model_file.streamlines, transfo)

    rng = np.random.RandomState(args.seed)
    if args.in_pickle:
        with open(args.in_pickle, 'rb') as infile:
            cluster_map = pickle.load(infile)
        reco_obj = RecoBundles(wb_streamlines,
                               cluster_map=cluster_map,
                               rng=rng,
                               verbose=args.verbose)
    else:
        reco_obj = RecoBundles(wb_streamlines,
                               clust_thr=args.tractogram_clustering_thr,
                               rng=rng,
                               verbose=args.verbose)

    if args.out_pickle:
        with open(args.out_pickle, 'wb') as outfile:
            pickle.dump(reco_obj.cluster_map, outfile)
    _, indices = reco_obj.recognize(ArraySequence(model_streamlines),
                                    args.model_clustering_thr,
                                    pruning_thr=args.pruning_thr,
                                    slr_num_threads=args.slr_threads)
    new_streamlines = wb_streamlines[indices]
    new_data_per_streamlines = wb_file.data_per_streamline[indices]
    new_data_per_points = wb_file.data_per_point[indices]

    if not args.no_empty or new_streamlines:
        sft = StatefulTractogram(new_streamlines, wb_file, Space.RASMM,
                                 data_per_streamline=new_data_per_streamlines,
                                 data_per_point=new_data_per_points)
        save_tractogram(sft, args.out_tractogram)


if __name__ == '__main__':
    main()
