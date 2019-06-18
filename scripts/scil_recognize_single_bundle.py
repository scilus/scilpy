#!/usr/bin/env python
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

from dipy.segment.bundles import RecoBundles
from dipy.tracking.streamline import transform_streamlines
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exists)


def buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Input tractogram filename')
    p.add_argument('in_model',
                   help='Model to use for recognition')
    p.add_argument('transformation',
                   help='4x4 transformation to bring the model onto \n'
                   'the input tractogram')
    p.add_argument('output_name',
                   help='Output tractogram filename')

    p.add_argument('--wb_clustering_thr', type=float, default=8,
                   help='Clustering threshold used for the whole brain '
                   '[%(default)smm]')
    p.add_argument('--model_clustering_thr', type=float, default=4,
                   help='Clustering threshold used for the model '
                   '[%(default)smm]')
    p.add_argument('--pruning_thr', type=float, default=6,
                   help='MDF threshold used for final streamlines selection '
                   '[%(default)smm]')

    p.add_argument('--slr_threads', type=int, default=None,
                   help='Number of threads for SLR [all]')
    p.add_argument('--seed', type=int, default=None,
                   help='Random number generator seed [%(default)s]')
    p.add_argument('--inverse', action='store_true',
                   help='Will use the inverse transformation')
    p.add_argument('--no_empty', action='store_true',
                   help='Do not write file if there is no streamline')

    group = p.add_mutually_exclusive_group()
    group.add_argument('--input_pickle',
                       help='Input pickle clusters map file.\n'
                       'Will override the wb_clustering_thr parameter')
    group.add_argument('--output_pickle',
                       help='Output pickle clusters map file.')

    p.add_argument('-v', action='store_true', dest='verbose',
                   help='Produce verbose output. [false]')

    add_overwrite_arg(p)

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.transformation])
    assert_outputs_exists(parser, args, [args.output_name])

    wb_file = nib.streamlines.load(args.in_tractogram)
    wb_streamlines = wb_file.streamlines
    model_file = nib.streamlines.load(args.in_model)

    # Default transformation source is expected to be ANTs
    transfo = np.loadtxt(args.transformation)
    if args.inverse:
        transfo = np.linalg.inv(np.loadtxt(args.transformation))

    model_streamlines = ArraySequence(
        transform_streamlines(model_file.streamlines, transfo))

    rng = np.random.RandomState(args.seed)
    if args.input_pickle:
        with open(args.input_pickle, 'rb') as infile:
            cluster_map = pickle.load(infile)
        reco_obj = RecoBundles(wb_streamlines,
                               cluster_map=cluster_map,
                               rng=rng,
                               verbose=args.verbose)
    else:
        reco_obj = RecoBundles(wb_streamlines,
                               clust_thr=args.wb_clustering_thr,
                               rng=rng,
                               verbose=args.verbose)

    if args.output_pickle:
        with open(args.output_pickle, 'wb') as outfile:
            pickle.dump(reco_obj.cluster_map, outfile)
    _, indices = reco_obj.recognize(model_streamlines,
                                    args.model_clustering_thr,
                                    pruning_thr=args.pruning_thr,
                                    slr_num_threads=args.slr_threads)
    recognized = [wb_streamlines[indice] for indice in indices]

    if not args.no_empty or recognized:
        out_tractogram = nib.streamlines.Tractogram(recognized,
                                                    affine_to_rasmm=np.eye(4))
        nib.streamlines.save(out_tractogram, args.output_name,
                             header=wb_file.header)


if __name__ == '__main__':
    main()
